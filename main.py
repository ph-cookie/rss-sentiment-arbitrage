import os
import sys
import re
import json
import logging
from typing import List, Dict, Any
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from feedgen.feed import FeedGenerator
import pytz
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

EXCLUDE_KEYWORDS = ["有料会員", "会員限定", "ログイン", "この記事は有料", "続きは有料", "プレミアム", "🔒"]
INTEREST_TEXT = "IT技術、プログラミング、人工知能、ガジェット、デジタル、データ分析、音楽、芸術"
THRESHOLD = 0.821
GEMINI_MODEL_NAME = 'gemini-3.1-flash-lite'

# キャッシュ設定
CACHE_FILE = "processed_urls.json"
MAX_CACHE_SIZE = 150 # 保持する過去のURLの最大件数

def load_cache() -> List[str]:
    """過去に処理したURLリストを読み込む"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                urls = json.load(f)
                logger.info(f"キャッシュ読込成功（既読URL: {len(urls)}件）")
                return urls
        except Exception as e:
            logger.warning(f"キャッシュの読み込みに失敗しました: {e}")
    else:
        logger.info("初回実行、またはキャッシュが存在しません。")
    return []

def save_cache(seen_urls: List[str]):
    """処理したURLリストを保存する（上限を設けてスライス）"""
    try:
        # 古いものを捨て、最新のMAX_CACHE_SIZE件だけ残す
        limited_urls = seen_urls[-MAX_CACHE_SIZE:]
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(limited_urls, f, ensure_ascii=False, indent=2)
        logger.info(f"キャッシュ保存成功（保持件数: {len(limited_urls)}件）")
    except Exception as e:
        logger.error(f"キャッシュの保存に失敗しました: {e}")

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

def fetch_free_articles(urls: List[str], seen_links: List[str], max_per_feed: int = 5) -> List[Any]:
    free_articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
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
            logger.error(f"URL取得エラー ({url}): {e}")
    return free_articles

def filter_articles_by_similarity(articles: List[Any], embedder: SentenceTransformer, interest_text: str, threshold: float) -> tuple[List[Dict], bool]:
    if not articles:
        return [], False

    interest_vector = embedder.encode([f"query: {interest_text}"])
    texts_to_embed = [f"passage: {getattr(entry, 'title', '')} {getattr(entry, 'summary', getattr(entry, 'description', ''))}" for entry in articles]
        
    logger.info("ベクトル化を一括実行中...")
    article_vectors = embedder.encode(texts_to_embed)
    
    scored_articles = []
    for idx, entry in enumerate(articles):
        sim = cosine_similarity(interest_vector, [article_vectors[idx]])[0][0]
        scored_articles.append({
            'entry': entry,
            'sim': float(sim),
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

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_ai_explanation(client: Any, original_title: str, summary: str) -> dict:
    prompt = f"""
以下のニュースの内容を詳細にまとめ、惹きつける分かりやすい「新しいタイトル」と、社会や読者に「どのような影響を与えるか」を中心に大学生向けに深く解説せよ。
指示：
- 【タイトル】と【解説】の見出しを必ず含めて出力すること。
- タイトルは30文字以内で、記事の要点とインパクトが伝わるものにすること。
- 解説は以下の構成とし、見出しは【】のみで記述すること（HTMLタグ不要）。
  【概要】
  (事実の要約)
  【背景・要因】
  (なぜ起きたか)
  【社会・読者への影響】
  (どのような影響があるか)
- 全体で400〜600文字程度とすること。
- 例え話は用いず、事実に基づいた客観的な影響を記述すること。
- Markdownの太字記号（アスタリスク2つ）は絶対に使用せず、強調はHTMLの <strong> タグを用いること。

出力フォーマット:
【タイトル】
(ここに新しいタイトル)

【解説】
【概要】
(概要本文)
【背景・要因】
(背景本文)
【社会・読者への影響】
(影響本文)

対象テキスト:
【{original_title}】 {summary}
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt
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
    gemini_key = os.environ.get("API_KEY1")
    hf_token = os.environ.get("HF_TOKEN1")
    
    if not gemini_key:
        logger.critical("API_KEY1 が設定されていない")
        sys.exit(1)
        
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    logger.info("0. 処理済みURLキャッシュの読み込み")
    seen_links = load_cache()

    logger.info("1. RSSフィードから記事取得")
    free_articles = fetch_free_articles(SOURCE_RSS_URLS, seen_links)
    logger.info(f"合計新規抽出数: {len(free_articles)}件")

    logger.info("2. モデルロードと類似度フィルタリング")
    embedder = SentenceTransformer('intfloat/multilingual-e5-small')
    target_articles, is_fallback = filter_articles_by_similarity(free_articles, embedder, INTEREST_TEXT, THRESHOLD)
    logger.info(f"処理対象: {len(target_articles)}件")

    logger.info("3. AI再構築とフィード生成")
    client = genai.Client(api_key=gemini_key)
    
    fg = FeedGenerator()
    fg.title('AI再構築フィード (興味外ニュースの平易化)')
    fg.link(href='https://github.com/', rel='alternate')
    fg.description('自身の興味領域外のニュースをAIがわかりやすく解説するカスタムフィード')
    fg.language('ja')

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
            logger.error(f"AI解説生成の最終エラー: {e}")
            ai_title = original_title
            ai_explanation = "解説の生成に失敗しました。"
            
        raw_link = getattr(entry, 'link', '').strip().split('#')[0]
        image_url = extract_image_url(entry, original_html)
        
        fe = fg.add_entry()
        fe.id(raw_link)
        fe.title(ai_title)
        fe.link(href=raw_link)
        
        img_html = f'<p><img src="{image_url}" style="max-width:100%; height:auto;" /></p>' if image_url else ""
        fallback_notice = "<p>※出力確保のための抽出記事</p>" if is_fallback else ""
        
        ai_exp_clean = re.sub(r'\n{2,}', '\n', ai_explanation).strip()
        ai_exp_html = ai_exp_clean.replace('\n', '<br>')
        
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
        custom_summary = f"興味類似度: {item['sim']:.3f}　|　ソース: {feed_title}"
        
        fe.description(custom_summary)
        fe.content(content=description_html, type='html')
        
        pub_parsed = getattr(entry, 'published_parsed', None)
        if pub_parsed:
            dt = datetime(*pub_parsed[:6])
            pub_date = pytz.utc.localize(dt).astimezone(pytz.timezone('Asia/Tokyo'))
        else:
            pub_date = datetime.now(pytz.timezone('Asia/Tokyo'))
            
        fe.pubDate(pub_date)
        
        if image_url:
            fe.enclosure(image_url, 0, 'image/jpeg')

        # 正常に処理した記事のURLをキャッシュリストに追加
        if raw_link not in seen_links:
            seen_links.append(raw_link)

        logger.info(f"処理完了: {ai_title[:15]}...")

    # キャッシュをJSONファイルに保存（次のActions実行へ引き継ぐ）
    save_cache(seen_links)

    fg.rss_file('rss.xml')
    logger.info("rss.xml 生成完了")

if __name__ == "__main__":
    main()