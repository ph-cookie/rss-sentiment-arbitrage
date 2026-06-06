import os
import sys
import time
import re
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from feedgen.feed import FeedGenerator
import pytz
from datetime import datetime
from huggingface_hub import login
from lxml import etree

# --- 設定 ---
# 取得元のRSS URL
SOURCE_RSS_URLS = [
    "https://rss.itmedia.co.jp/rss/2.0/business.xml",   # ITmedia ビジネスオンライン
    "https://prtimes.jp/index.rdf",                     # PR TIMES
    "https://assets.wor.jp/rss/rdf/nikkei/news.rdf",    # 日経新聞（テスト用）
    "https://feeds.japan.cnet.com/rss/cnet/all.rdf",    # CNET Japan
    "https://news.yahoo.co.jp/rss/topics/domestic.xml", # Yahoo 国内
    "https://news.yahoo.co.jp/rss/topics/world.xml",    # Yahoo 国際
    "https://news.yahoo.co.jp/rss/topics/business.xml", # Yahoo ビジネス
    "https://feeds.bbci.co.uk/japanese/rss.xml"         # BBC 日本語版
]

# 有料記事・ログイン必須記事を示すキーワード群
EXCLUDE_KEYWORDS = [
    "有料会員", "会員限定", "ログイン", 
    "この記事は有料", "続きは有料", "プレミアム", "🔒"
]

# 興味関心を定義（これに似ていない記事を抽出）
INTEREST_TEXT = "IT技術、プログラミング、人工知能、ガジェット、デジタル、データ分析、音楽、芸術"

# 類似度の閾値（-1.0 〜 1.0）。この数値以下の記事を「興味外」と判定
THRESHOLD = 0.821

# -----------
def main():
    gemini_key = os.environ.get("API_KEY1")
    hf_token = os.environ.get("HF_TOKEN1")
    
    if not gemini_key:
        print("エラー: API_KEY1 (Gemini APIキー) が設定されていません")
        sys.exit(1)
    
    if hf_token:
        login(token=hf_token)
    
    print("1. 各RSSフィードから無料記事を各5件ずつ取得中...")
    free_articles = []
    
    for url in SOURCE_RSS_URLS:
        try:
            feed = feedparser.parse(url)
            collected_count = 0
            
            for entry in feed.entries:
                if collected_count >= 5:
                    break
                    
                summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
                text_to_check = f"{entry.title} {summary}"
                
                # 除外キーワードが含まれているか判定
                is_paid = any(keyword in text_to_check for keyword in EXCLUDE_KEYWORDS)
                
                if not is_paid:
                    free_articles.append(entry)
                    collected_count += 1
                    
            print(f"  - 取得完了: {url} (無料記事: {collected_count}件)")
                    
        except Exception as e:
            print(f"URL取得エラー ({url}): {e}")
            
    print(f"-> 全サイト合計で {len(free_articles)}件 の無料記事を抽出。")

    print("2. ローカルAIモデルをロード中 (ベクトル化)...")
    # token引数を環境変数から渡す
    embedder = SentenceTransformer('intfloat/multilingual-e5-small', token=hf_token)
    interest_vector = embedder.encode(["query: " + INTEREST_TEXT])
    
    target_articles = []
    
    print("3. 類似度計算とフィルタリングを実行中...")
    scored_articles = []
    for entry in free_articles:
        summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
        text_to_embed = "passage: " + f"{entry.title} {summary}"
        
        article_vector = embedder.encode([text_to_embed])
        sim = cosine_similarity(interest_vector, article_vector)[0][0]
        scored_articles.append({'entry': entry, 'sim': sim, 'summary': summary})
        
        print(f"[{sim:.3f}] {entry.title[:30]}...")

    # 閾値以下の記事を抽出
    target_articles = [item for item in scored_articles if item['sim'] < THRESHOLD]
    is_fallback = False
    
    # 0件だった場合の強制抽出（相対的に類似度が低い上位3件を取得）
    if len(target_articles) == 0 and len(scored_articles) > 0:
        print("-> 閾値以下の記事が0件のため、類似度が低い上位3件を強制抽出する。")
        scored_articles.sort(key=lambda x: x['sim'])
        target_articles = scored_articles[:3]
        is_fallback = True

    print(f"-> 処理対象: {len(target_articles)}件")

    print("4. テキスト再構築中...")
    genai.configure(api_key=gemini_key)
    llm_model = genai.GenerativeModel('gemini-3.1-flash-lite') 
    
    fg = FeedGenerator()
    fg.title('AI再構築フィード (興味外ニュースの平易化)')
    fg.link(href='https://github.com/', rel='alternate')
    fg.description('自身の興味領域外のニュースをAIがわかりやすく解説するカスタムフィード')
    fg.language('ja')

    for item in target_articles:
        entry = item['entry']
        summary = item['summary']
        
        # HTML抽出ロジックの簡略化
        if hasattr(entry, 'content') and len(entry.content) > 0:
            original_html = entry.content[0].get('value', summary)
        else:
            original_html = summary
            
        fallback_notice = "※この記事は類似度閾値を満たしませんでしたが、出力確保のため抽出されました。" if is_fallback else ""
        
        prompt = f"""
        以下のニュースの内容を簡潔にまとめ、それが社会や読者に「どのような影響を与えるか」を中心にして、大学生向けにわかりやすく解説してください。
        例え話は用いず、事実に基づいた客観的な影響を記述してください。
        HTMLタグは含めず、プレーンテキストで出力してください。
        
        対象テキスト: 
        【{entry.title}】 {summary}
        """
        
        try:
            response = llm_model.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                ai_explanation = response.text
                ai_explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', ai_explanation)
                ai_explanation = re.sub(r'\*(.*?)\*', r'\1', ai_explanation)
                ai_explanation = ai_explanation.replace('* ', '・')
            else:
                ai_explanation = "AIによる解説が生成されませんでした（フィルターブロック等の理由）。"
        except Exception as e:
            ai_explanation = f"AI解説の生成エラー: {e}"

        raw_link = entry.link.strip().split('#')[0]
        
        fe = fg.add_entry()
        fe.id(raw_link)
        fe.title(f"{entry.title}")
        fe.link(href=raw_link)
        
        image_url = ""
        if hasattr(entry, 'links'):
            for link in entry.links:
                if 'image' in link.get('type', ''):
                    image_url = link.get('href', '')
                    break
        if not image_url and hasattr(entry, 'media_content') and len(entry.media_content) > 0:
            image_url = entry.media_content[0].get('url', '')
        if not image_url and hasattr(entry, 'media_thumbnail') and len(entry.media_thumbnail) > 0:
            image_url = entry.media_thumbnail[0].get('url', '')
        if not image_url and 'src=' in original_html:
            img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', original_html)
            if img_match:
                image_url = img_match.group(1)
        
        img_html = ""
        if image_url:
            fe.enclosure(image_url, 0, 'image/jpeg')
            img_html = f'<p><img src="{image_url}" style="max-width:100%; height:auto;" /></p>'
        
        # 修正：興味度スコアをAI書き換え本文の前に配置
        description_html = f"""
        <p><small style="color:gray;">興味類似度スコア: {item['sim']:.3f}</small></p>
        {fallback_notice}
        {img_html}
        <h3>AI書換え本文</h3>
        <p>{ai_explanation.replace(chr(10), '<br>')}</p>
        <hr>
        <h3>元の記事</h3>
        {original_html}
        """
        
        # summaryが空っぽの場合のフォールバックを用意（リスト表示崩れ防止）
        safe_summary = summary if summary.strip() else "記事の概要が提供されていません。"
        fe.description(safe_summary)
        
        # lxml.etree.CDATA を使ってHTMLを正しくラップする（HTMLレンダリング有効化）
        fe.content(etree.CDATA(description_html))
        
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6])
            pub_date = pytz.utc.localize(dt).astimezone(pytz.timezone('Asia/Tokyo')).isoformat()
        else:
            pub_date = datetime.now(pytz.timezone('Asia/Tokyo')).isoformat()
        
        fe.pubDate(pub_date)

        print(f"  - 処理完了: {entry.title[:15]}... (待機中)")
        time.sleep(4) # 429防止

    print("5. XMLファイルを生成中...")
    fg.rss_file('rss.xml')
    print("完了: rss.xml 正常に生成。")

if __name__ == "__main__":
    main()