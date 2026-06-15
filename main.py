import os
import sys
import re
import json
import logging
import mimetypes
import time
import socket
from urllib.parse import urlparse
from urllib.error import URLError
from typing import List, Dict, Any
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from feedgen.feed import FeedGenerator
import pytz
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ネットワーク通信のグローバルタイムアウトを設定（ハングアップ防止）
socket.setdefaulttimeout(15)

# 取得元のRSS URL
SOURCE_RSS_URLS = [
    "https://rss.itmedia.co.jp/rss/2.0/business.xml",   # ITmedia ビジネスオンライン
    "https://prtimes.jp/index.rdf",                     # PR TIMES
    "https://assets.wor.jp/rss/rdf/nikkei/news.rdf",    # 日経新聞
    "https://feeds.japan.cnet.com/rss/cnet/all.rdf",    # CNET Japan
    "https://news.yahoo.co.jp/rss/topics/domestic.xml", # Yahoo 国内
    "https://news.yahoo.co.jp/rss/topics/world.xml",    # Yahoo 国際
    "https://news.yahoo.co.jp/rss/topics/business.xml", # Yahoo ビジネス
    "https://feeds.bbci.co.uk/japanese/rss.xml"         # BBC 日本語版
]

EXCLUDE_KEYWORDS = ["有料会員", "会員限定", "ログイン", "この記事は有料", "続きは有料", "プレミアム", "🔒"]
INTEREST_TEXTS = [
    "IT技術、プログラミング、人工知能、データ分析",
    "ガジェット、情報、デジタル",
    "音楽、芸術、ゲーム、推し活",
    "旅、自然"
]

THRESHOLD = 0.820
GEMINI_MODEL_NAME = 'gemini-3.1-flash-lite'

# キャッシュ設定
CACHE_FILE = "processed_urls.json"
MAX_CACHE_SIZE = 500 # 保持する過去のURLの最大件数

def get_mime_type(url: str) -> str:
    """URLからMIMEタイプを推測する（クエリパラメータを無視）"""
    if not url:
        return 'image/jpeg'
    parsed_url = urlparse(url)
    mime_type, _ = mimetypes.guess_type(parsed_url.path)
    return mime_type or 'image/jpeg'

def load_cache() -> tuple[List[str], List[Dict]]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    logger.info("旧形式のキャッシュを検出。フォーマットを移行します。")
                    return data, []
                
                seen_urls = data.get("seen_urls", [])
                last_run_articles = data.get("last_run_articles", [])
                logger.info(f"キャッシュ読込成功（既読URL: {len(seen_urls)}件, 前回記事: {len(last_run_articles)}件）")
                return seen_urls, last_run_articles
        except Exception as e:
            logger.warning(f"キャッシュの読み込みに失敗: {e}")
    else:
        logger.info("初回実行、またはキャッシュが存在しません。")
    return [], []

def save_cache(seen_urls: List[str], current_run_articles: List[Dict]):
    try:
        # 古いものを捨て、最新のMAX_CACHE_SIZE件だけ残す
        limited_urls = seen_urls[-MAX_CACHE_SIZE:]
        data = {
            "seen_urls": limited_urls,
            "last_run_articles": current_run_articles
        }
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"キャッシュ保存成功（URL保持: {len(limited_urls)}件, 保存記事: {len(current_run_articles)}件）")
    except Exception as e:
        logger.error(f"キャッシュの保存失敗: {e}")

def extract_image_url(entry: Any, original_html: str) -> str:
    links = getattr(entry, 'links', [])
    image_url = next((link.get('href') for link in links if 'image' in link.get('type', '')), '')
    
    if not image_url:
        for attr in ['media_content', 'media_thumbnail']:
            media = getattr(entry, attr, [])
            if media:
                image_url = media[0].get('url', '')
                break
                
    if not image_url and 'src=' in original_html:
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', original_html)
        if img_match:
            image_url = img_match.group(1)
            
    return image_url

def fetch_free_articles(urls: List[str], seen_links: List[str], max_per_feed: int = 10) -> List[Any]:
    free_articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            
            if getattr(feed, 'bozo', 0) == 1:
                exception = getattr(feed, 'bozo_exception', 'Unknown error')
                if isinstance(exception, URLError) and isinstance(exception.reason, socket.timeout):
                    logger.error(f"URL取得タイムアウト ({url})")
                    continue
                else:
                    logger.warning(f"フィード解析警告 ({url}): {exception}")
            
            feed_title = getattr(feed.feed, 'title', url)
            collected = 0
            excluded_count = 0
            skipped_count = 0 # 既読スキップのカウント
            
            for entry in feed.entries:
                if collected >= max_per_feed:
                    break
                
                # 既読（キャッシュ済み）のURLはスキップ
                raw_link = getattr(entry, 'link', '').strip().split('#')[0]
                if raw_link in seen_links:
                    skipped_count += 1
                    continue
                    
                summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
                title = getattr(entry, 'title', '')
                text_to_check = f"{title} {summary}"
                
                excluded_kw = next((kw for kw in EXCLUDE_KEYWORDS if kw in text_to_check), None)
                if excluded_kw:
                    logger.info(f"除外 ({excluded_kw}): {title[:20]}...")
                    excluded_count += 1
                    continue
                
                entry['feed_title'] = feed_title
                free_articles.append(entry)
                collected += 1
                
            if excluded_count == 0 and skipped_count == 0 and collected == 0:
                logger.info(f"処理対象記事なし: {url}")
            else:
                logger.info(f"取得完了: {url} (新規: {collected}件 / 除外: {excluded_count}件 / 既読スキップ: {skipped_count}件)")
                
        except Exception as e:
            logger.error(f"URL取得予期せぬエラー ({url}): {e}")
            
    return free_articles

def filter_articles_by_similarity(articles: List[Any], embedder: SentenceTransformer, interest_texts: List[str], threshold: float) -> tuple[List[Dict], bool]:
    if not articles:
        return [], False

    try:
        # 複数の興味関心ベクトルを一括生成
        interest_queries = [f"query: {text}" for text in interest_texts]
        interest_vectors = embedder.encode(interest_queries)
        
        texts_to_embed = [f"passage: {getattr(entry, 'title', '')} {getattr(entry, 'summary', getattr(entry, 'description', ''))}" for entry in articles]
            
        logger.info(f"ベクトル化を一括実行中... (記事: {len(articles)}件, 興味クラスタ: {len(interest_texts)}件)")
        article_vectors = embedder.encode(texts_to_embed)
        
        scored_articles = []
        for idx, entry in enumerate(articles):
            # 各記事に対して、全ての興味クラスタとの類似度を計算し、その最大値を取得
            sims = cosine_similarity([article_vectors[idx]], interest_vectors)[0]
            max_sim = float(max(sims))
            
            scored_articles.append({
                'entry': entry,
                'sim': max_sim,
                'summary': getattr(entry, 'summary', getattr(entry, 'description', ''))
            })

        target_articles = [item for item in scored_articles if item['sim'] < threshold]
        is_fallback = False
        
        if not target_articles and scored_articles:
            logger.warning("閾値以下の記事が0件。類似度下位3件を強制抽出。")
            scored_articles.sort(key=lambda x: x['sim'])
            target_articles = scored_articles[:3]
            is_fallback = True
            
        return target_articles, is_fallback
    except Exception as e:
        logger.error(f"ベクトル化処理中に致命的なエラーが発生: {e}")
        return [], False

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=10, max=65))
def generate_ai_explanation(client: Any, original_title: str, summary: str) -> dict:
    prompt = f"""
以下のニュースの内容を読み、IT技術やデータ分析、論理的思考を好む読者が「構造的な面白さ」を感じて思わず読みたくなる「新しいタイトル」と、深掘りした「解説」を作成せよ。

【タイトルの作成ルール】
- 形式は「[具体的な事実]：[それが引き起こすシステム・社会構造への影響]」の2部構成とすること。
- コンピュータやIT用語への「無理な例え話（アナロジー・比喩）」は絶対に使用しないこと。
- そのニュースが持つ「力学の変化」「エコシステムの変容」「データの動き」「心理的パラドックス」などのメタ視点を提示すること。
- 35文字以内で、簡潔かつ鋭い表現にすること。

【解説の作成ルール】
- [対象テキスト]にある情報のみを情報源とし、事実関係を勝手に変更・推測しないこと。
- 以下の構成とし、見出しは【】のみで記述すること（HTMLタグ不要）。
  【概要】
  [対象テキスト]の事実要約
  【背景・要因】
  [対象テキスト]から読み取れる背景や原因を抽出。
  なぜ起きたか。検索機能やモデルの知識を活用し、技術的仕組み、ビジネスモデルの構造、歴史的経緯など「普遍的な知識」で深掘りする。ただし、現在の政治家名や役職など、最新の時事情報を勝手に断定・推測することは厳禁。
  【社会・読者への影響】
  マクロな視点での波及効果。
- 外部知識（モデルの学習）による補足は一切行わず、テキストに存在しない情報は「情報不足のため不明」とすること。
- 強調箇所はHTMLの <strong> タグを用いること。Markdownの太字記号は絶対に使用しないこと。

出力フォーマット:
【タイトル】
(ここに新しいタイトル)

【解説】
【概要】
(概要本文)
\n\n
【背景・要因】
(背景本文)
\n\n
【社会・読者への影響】
(影響本文)

対象テキスト:
【{original_title}】 {summary}
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            tools=[{"google_search": {}}]
        )
    )
    
    result = {"title": original_title, "explanation": "AIによる解説が生成されませんでした（フィルタブロック等）。"}
    
    if response.text:
        text = response.text.strip()
        title_match = re.search(r'【タイトル】\s*(.*?)\s*【解説】', text, re.DOTALL)
        explanation_match = re.search(r'【解説】\s*(.*)', text, re.DOTALL)
        
        if title_match and explanation_match:
            result["title"] = title_match.group(1).strip()
            result["explanation"] = explanation_match.group(1).strip()
        else:
            result["explanation"] = text
            
    return result

def main():
    try:
        gemini_key = os.environ.get("API_KEY1")
        hf_token = os.environ.get("HF_TOKEN1")
        
        if not gemini_key:
            logger.critical("API_KEY1 が未設定")
            sys.exit(1)
            
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        logger.info("0. 処理済みURLキャッシュの読み込み")
        seen_links, last_run_articles = load_cache()

        logger.info("1. RSSフィードから記事取得")
        free_articles = fetch_free_articles(SOURCE_RSS_URLS, seen_links)
        logger.info(f"合計新規抽出数: {len(free_articles)}件")

        if not free_articles:
            logger.info("新規処理対象の記事がないため、キャッシュの更新・RSS再構築のみ行います。")
            target_articles = []
            is_fallback = False
        else:
            logger.info("2. モデルロードと類似度フィルタリング")
            try:
                embedder = SentenceTransformer('intfloat/multilingual-e5-small')
                target_articles, is_fallback = filter_articles_by_similarity(free_articles, embedder, INTEREST_TEXTS, THRESHOLD)
            except Exception as e:
                logger.error(f"モデルロードに失敗: {e}")
                target_articles, is_fallback = [], False

        logger.info(f"処理対象: {len(target_articles)}件")
        
        logger.info("3. AI再構築とフィード生成")
        client = genai.Client(api_key=gemini_key)
        current_run_articles = []

        for item in target_articles:
            entry = item['entry']
            summary = item['summary']
            original_title = getattr(entry, 'title', '')
            
            content_obj = getattr(entry, 'content', [])
            original_html = content_obj[0].get('value', summary) if content_obj else summary
            
            try:
                ai_result = generate_ai_explanation(client, original_title, summary)
                ai_title = ai_result["title"]
                ai_explanation = ai_result["explanation"]
            except Exception as e:
                if "429" in str(e):
                    logger.error(f"APIレートリミット到達 (429): {e}")
                else:
                    logger.error(f"AI解説生成に失敗 (API起因等): {e}")
                
                ai_title = original_title
                ai_explanation = f"【概要】\n{summary}\n\n【構造的背景】\n情報不足または生成エラーのため不明\n\n【社会・読者への影響】\n情報不足または生成エラーのため不明"
                
            raw_link = getattr(entry, 'link', '').strip().split('#')[0]
            image_url = extract_image_url(entry, original_html)
            
            fallback_notice = "<p>※出力確保のための抽出記事</p>" if is_fallback else ""
            
            ai_exp_clean = re.sub(r'\n{2,}', '\n', ai_explanation).strip()
            ai_exp_html = ai_exp_clean.replace('\n', '<br>')
            
            img_html = f'<p><img src="{image_url}" style="max-width:100%; height:auto;" /></p>' if image_url else ""
            
            description_html = f"""
            <p style="color:gray; font-size: small;">
            ・元タイトル「{original_title}」<br>
            ・興味類似度スコア: {item['sim']:.3f}
            </p>
            {fallback_notice}
            <h3>AI書換え本文</h3>
            {img_html}
            <p>{ai_exp_html}</p>
            <hr>
            <h3>元の記事</h3>
            {original_html}
            """
            
            feed_title = entry.get('feed_title', '不明なソース')
            custom_summary = f"興味類似度: {item['sim']:.3f} | ソース: {feed_title}"
            
            pub_parsed = getattr(entry, 'published_parsed', None)
            if pub_parsed:
                dt = datetime(*pub_parsed[:6])
                pub_date_iso = pytz.utc.localize(dt).astimezone(pytz.timezone('Asia/Tokyo')).isoformat()
            else:
                pub_date_iso = datetime.now(pytz.timezone('Asia/Tokyo')).isoformat()
                
            article_data = {
                "id": raw_link,
                "title": ai_title,
                "link": raw_link,
                "description": custom_summary,
                "content": description_html,
                "pubDate": pub_date_iso,
                "enclosure": image_url
            }
            current_run_articles.append(article_data)

            # 正常に処理した記事のURLをキャッシュリストに追加
            if raw_link not in seen_links:
                seen_links.append(raw_link)

            logger.info(f"処理完了: {ai_title[:15]}...")

            # 制限15RPM -> 60秒÷15回=最低4秒
            time.sleep(15)

        # 重複を排除しつつ、今回と過去の記事を結合
        unique_articles = {art["id"]: art for art in (current_run_articles + last_run_articles)}
        all_feed_articles = list(unique_articles.values())
        
        # 日付が新しい順（降順）にソート
        all_feed_articles.sort(key=lambda x: x["pubDate"], reverse=True)
        
        # 常に最新の30件のみを維持する（RSSフィードの肥大化防止）
        MAX_FEED_ITEMS = 30
        all_feed_articles = all_feed_articles[:MAX_FEED_ITEMS]
        
        fg = FeedGenerator()
        fg.title('AI再構築フィード (興味外ニュースの平易化)')
        fg.link(href='https://github.com/', rel='alternate')
        fg.description('自身の興味領域外のニュースをAIがわかりやすく解説するカスタムフィード')
        fg.language('ja')

        for art in all_feed_articles:
            fe = fg.add_entry()
            fe.id(art["id"])
            fe.title(art["title"])
            fe.link(href=art["link"])
            fe.description(art["description"])
            fe.content(content=art["content"], type='html')
            fe.pubDate(datetime.fromisoformat(art["pubDate"]))
            
            if art.get("enclosure"):
                mime_type = get_mime_type(art["enclosure"])
                fe.enclosure(art["enclosure"], 0, mime_type)

        save_cache(seen_links, all_feed_articles)

        try:
            fg.rss_file('rss.xml')
            logger.info(f"rss.xml 生成完了 (出力件数: {len(all_feed_articles)}件)")
            
            # index.html の自動生成
            html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI再構築フィード</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2rem; color: #333; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
        .rss-link {{ display: inline-block; background: #ee802f; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 1rem; }}
        .rss-link:hover {{ background: #c66a26; }}
    </style>
</head>
<body>
    <h1>AI再構築フィード (興味外ニュースの平易化)</h1>
    <p>このページは、自動生成されたカスタムRSSフィードのホスティングサイトです。</p>
    <p>フィルターバブルを打破するため、ユーザーの関心領域外のニュースをAI（Gemini）が大学生向けに平易化・構造化して配信しています。</p>
    <p>最終更新: {datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} (JST)</p>
    
    <h2>利用方法</h2>
    <p>お使いのRSSリーダー（Feedly, Inoreaderなど）に以下のリンクを登録してください。</p>
    <a href="rss.xml" class="rss-link">RSSフィード (rss.xml) を取得</a>
    
    <p style="margin-top: 3rem; font-size: 0.8rem; color: #666;">
        Powered by GitHub Actions & Gemini API<br>
        <a href="https://github.com/ph-cookie/rss-sentiment-arbitrage">GitHub Repository</a>
    </p>
</body>
</html>
"""
            with open('index.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info("index.html 生成完了")

        except Exception as e:
            logger.error(f"ファイルの書き出しに失敗しました: {e}")
            raise

    except Exception as e:
        logger.critical(f"予期せぬ致命的なエラーにより処理が中断されました: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()