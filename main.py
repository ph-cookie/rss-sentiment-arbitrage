import os
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from feedgen.feed import FeedGenerator
import pytz
from datetime import datetime
import time
import re

# --- 設定 ---
# 取得元のRSS URL
SOURCE_RSS_URL = "https://prtimes.jp/index.rdf"

# あなたの「興味関心」を定義するテキスト（これに似ていない記事を抽出します）
INTEREST_TEXT = "IT技術、プログラミング、人工知能、ガジェット、デジタル、データ分析、音楽、芸術"

# 類似度の閾値（-1.0 〜 1.0）。この数値以下の記事を「興味外」と判定する
THRESHOLD = 0.80
# -----------

def main():
    print("1. RSSフィードを取得中...")
    feed = feedparser.parse(SOURCE_RSS_URL)
    articles = feed.entries[:15] # 処理時間を考慮し最新15件に制限

    print("2. ローカルAIモデルをロード中 (ベクトル化)...")
    # token引数を追加し、環境変数からHF_TOKENを渡す
    hf_token = os.environ.get("HF_TOKEN")
    embedder = SentenceTransformer('intfloat/multilingual-e5-small')
    interest_vector = embedder.encode(["query: " + INTEREST_TEXT])

    target_articles = []
    
    print("3. 類似度計算とフィルタリングを実行中...")
    for entry in articles:
        summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
        text_to_embed = "passage: " + f"{entry.title} {summary}"
        
        article_vector = embedder.encode([text_to_embed])
        sim = cosine_similarity(interest_vector, article_vector)[0][0]
        
        print(f"[{sim:.3f}] {entry.title}")
        
        if sim < THRESHOLD:
            target_articles.append({'entry': entry, 'sim': sim, 'summary': summary})
            
    print(f"-> {len(articles)}件中、{len(target_articles)}件を「興味外（要解説）」と判定。")

    print("4. Gemini API テキスト再構築中...")
    genai.configure(api_key=os.environ["API_KEY"])
    llm_model = genai.GenerativeModel('gemini-3.1-flash-lite')
    
    # 新規RSSフィードの初期化
    fg = FeedGenerator()
    fg.title('AI再構築フィード (興味外ニュースの平易化)')
    fg.link(href='https://github.com/', rel='alternate') # 公開後は自身のGitHub PagesのURLに変更
    fg.description('自身の興味領域外のニュースをAIがわかりやすく解説するカスタムフィード')
    fg.language('ja')

    for item in target_articles:
        entry = item['entry']
        summary = item['summary']
        
        # 元のHTMLコンテンツ取得
        original_html = ''
        if hasattr(entry, 'content'):
            original_html = entry.content[0].value
        else:
            original_html = summary
        
        prompt = f"""
        以下のニュースを「{INTEREST_TEXT}」といった概念や文脈に例えて、大学生向けに柔らかく書き換えてください。
        その際、専門用語には直後に括弧書きで簡潔な解説を補足してください。例：インフレ（=物価が継続的に上がる状態）
        
        【重要】
        ・HTMLタグは含めない
        ・太字（**）や斜体（*）などのMarkdown装飾は一切使用せず、完全なプレーンテキストで出力する
        ・箇条書きをする場合は「*」ではなく「・」を使用する
        
        対象テキスト: {summary}
        """
        
        try:
            response = llm_model.generate_content(prompt)
            ai_explanation = response.text
            
            # マークダウン除去
            # AIが出力しがちな **太字** や *斜体* の記号を消し、箇条書きの * を ・ に変換
            ai_explanation = re.sub(r'\*\*(.*?)\*\*', r'\1', ai_explanation)
            ai_explanation = re.sub(r'\*(.*?)\*', r'\1', ai_explanation)
            ai_explanation = ai_explanation.replace('* ', '・')

        except Exception as e:
            ai_explanation = f"AI解説の生成に失敗: {e}"

        # URLのクリーンアップ（空白・改行・不要なフラグメントを除去）
        raw_link = entry.link.strip().split('#')[0]
        
        fe = fg.add_entry()
        fe.id(raw_link)
        fe.title(f"[AI解説] {entry.title}")
        
        # feedgenの仕様に合わせ、hrefを辞書型または直接指定で確実に渡す
        fe.link(href=raw_link)
        
        description_html = f"""
        <h3>AI書換え本文</h3>
        <p>{ai_explanation.replace(chr(10), '<br>')}</p>
        <hr>
        <h3>元の記事</h3>
        {original_html}
        <hr>
        <p><small>興味類似度スコア: {item['sim']:.3f}</small></p>
        """

        # リーダーアプリ対応：
        # descriptionにはプレーンテキスト（一覧用）、contentにHTML（詳細用）をセットする
        fe.description(summary) 
        fe.content(description_html)
        
        # RSSの公開日時（pubDate）の引き継ぎ
        # feedparserの published_parsed を取得してdatetime型に変換
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            # タイムゾーンを持たないUTCの日時を作成
            dt = datetime(*entry.published_parsed[:6])
            # UTCとして認識させた後、日本時間（JST）に変換
            pub_date = pytz.utc.localize(dt).astimezone(pytz.timezone('Asia/Tokyo'))
        else:
            # 取得できない場合のみ現在時刻を使用
            pub_date = datetime.now(pytz.timezone('Asia/Tokyo'))
        
        fe.pubDate(pub_date)

        # APIのレートリミット対策
        # RPM (1分あたりのリクエスト数): 15 回
        # TPM (1分あたりのトークン数): 100万 トークン
        # RPD (1日あたりのリクエスト数): 1,500 回
        print(f"  - 処理完了: {entry.title[:15]}... (待機中)")
        time.sleep(4) # 429防止

    print("5. XMLファイルを生成中...")
    # 'rss.xml' として出力
    fg.rss_file('rss.xml')
    print("完了: rss.xml 正常に生成。")

if __name__ == "__main__":
    main()
