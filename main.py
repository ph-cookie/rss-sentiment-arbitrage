import os
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from feedgen.feed import FeedGenerator
import pytz
from datetime import datetime

# --- 設定 ---
# 取得元のRSS URL (例として日経やロイターなどを想定)
SOURCE_RSS_URL = "https://assets.wor.jp/rss/rdf/nikkei/news.rdf" 

# あなたの「興味関心」を定義するテキスト（これに似ていない記事を抽出します）
INTEREST_TEXT = "IT技術、プログラミング、人工知能、金融市場、ガジェット"

# 類似度の閾値（-1.0 〜 1.0）。この数値以下の記事を「興味外」と判定する
THRESHOLD = 0.35 
# -----------

def main():
    print("1. RSSフィードを取得中...")
    feed = feedparser.parse(SOURCE_RSS_URL)
    articles = feed.entries[:15] # 処理時間を考慮し最新15件に制限

    print("2. ローカルAIモデルをロード中 (ベクトル化)...")
    # 軽量で多言語対応のモデルを使用（APIを使用せずローカルで完結）
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    interest_vector = embedder.encode([INTEREST_TEXT])

    target_articles = []
    
    print("3. 類似度計算とフィルタリングを実行中...")
    for entry in articles:
        # タイトルと概要を結合してベクトル化
        summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
        text_to_embed = f"{entry.title} {summary}"
        
        article_vector = embedder.encode([text_to_embed])
        # コサイン類似度の計算
        sim = cosine_similarity(interest_vector, article_vector)[0][0]
        
        if sim < THRESHOLD:
            target_articles.append({'entry': entry, 'sim': sim, 'summary': summary})
            
    print(f"-> {len(articles)}件中、{len(target_articles)}件を「興味外（要解説）」と判定しました。")

    print("4. Gemini APIによるテキスト再構築を実行中...")
    genai.configure(api_key=os.environ["API_KEY"])
    # 無料枠で利用可能、かつ高速なflashモデルを指定
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 新規RSSフィードの初期化
    fg = FeedGenerator()
    fg.title('AI再構築フィード (興味外ニュースの平易化)')
    fg.link(href='https://github.com/', rel='alternate') # 公開後は自身のGitHub PagesのURLに変更
    fg.description('自身の興味領域外のニュースをAIがわかりやすく解説するカスタムフィード')
    fg.language('ja')

    for item in target_articles:
        entry = item['entry']
        summary = item['summary']
        
        # 元のHTMLコンテンツの取得（フルコンテンツがあれば優先）
        original_html = ''
        if hasattr(entry, 'content'):
            original_html = entry.content[0].value
        else:
            original_html = summary
        
        # LLMへのプロンプト指示（大学生向け・用語解説）
        prompt = f"""
        以下のニュースを、大学生が理解しやすい柔らかい表現に書き換えてください。
        その際、専門用語には直後に括弧書きで簡潔な解説を補足してください。例：インフレ（=物価が継続的に上がる状態）
        HTMLタグは含めず、プレーンテキストで出力してください。
        
        対象テキスト: {summary}
        """
        
        try:
            response = llm_model.generate_content(prompt)
            ai_explanation = response.text
        except Exception as e:
            ai_explanation = f"AI解説の生成に失敗しました: {e}"

        fe = fg.add_entry()
        fe.id(entry.link)
        fe.title(f"[AI解説] {entry.title}")
        fe.link(href=entry.link)
        
        # AIのテキストと元のHTMLを結合
        description_html = f"""
        <h3>AI書き換え本文</h3>
        <p>{ai_explanation.replace(chr(10), '<br>')}</p>
        <hr>
        <h3>元の記事</h3>
        {original_html}
        <hr>
        <p><small>興味類似度スコア: {item['sim']:.3f}</small></p>
        """
        fe.description(description_html)
        
        now = datetime.now(pytz.timezone('Asia/Tokyo'))
        fe.pubDate(now)

    print("5. XMLファイルを生成中...")
    # 'rss.xml' として出力
    fg.rss_file('rss.xml')
    print("完了: rss.xml が正常に生成されました。")

if __name__ == "__main__":
    main()