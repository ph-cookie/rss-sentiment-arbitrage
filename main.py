import os
import sys
import re
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

def fetch_free_articles(urls: List[str], max_per_feed: int = 5) -> List[Any]:
    free_articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            collected = 0
            for entry in feed.entries:
                if collected >= max_per_feed:
                    break
                summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
                title = getattr(entry, 'title', '')
                text_to_check = f"{title} {summary}"
                
                if not any(kw in text_to_check for kw in EXCLUDE_KEYWORDS):
                    free_articles.append(entry)
                    collected += 1
            logger.info(f"取得完了: {url} (無料記事: {collected}件)")
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
def generate_ai_explanation(client: Any, title: str, summary: str) -> str:
    prompt = f"""
以下のニュースの内容を詳細にまとめ、惹きつける分かりやすい「新しいタイトル」と、社会や読者に「どのような影響を与えるか」を中心に大学生向けに深く解説せよ。
指示：
- 【タイトル】と【解説】の見出しを必ず含めて出力すること。
- タイトルは30文字以内で、記事の要点とインパクトが伝わるものにすること。
- 文字数は400〜600文字程度とし、具体的な背景や要因を含めて記述すること。
- 例え話は用いず、事実に基づいた客観的な影響を記述すること。
- Markdownの太字記号（**）は絶対に使用しないこと。

出力フォーマット:
【タイトル】
(ここに新しいタイトル)

【解説】
(ここに解説本文)

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

    logger.info("1. RSSフィードから記事取得")
    free_articles = fetch_free_articles(SOURCE_RSS_URLS)
    logger.info(f"合計抽出数: {len(free_articles)}件")

    logger.info("2. モデルロードと類似度フィルタリング")
    embedder = SentenceTransformer('intfloat/multilingual-e5-small', token=hf_token)
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
        fe.title(ai_title) # 生成された新しいタイトルを適用
        fe.link(href=raw_link)
        
        img_html = f'<p><img src="{image_url}" style="max-width:100%; height:auto;" /></p>' if image_url else ""
        fallback_notice = "<p>※出力確保のための抽出記事</p>" if is_fallback else ""
        
        description_html = f"""
        <p><small style="color:gray;">元のタイトル: {original_title} | 興味類似度スコア: {item['sim']:.3f}</small></p>
        {fallback_notice}
        {img_html}
        <h3>AI書換え本文</h3>
        <p>{ai_explanation.replace(chr(10), '<br>')}</p>
        <hr>
        <h3>元の記事</h3>
        {original_html}
        """
        
        safe_summary = summary if summary.strip() else "概要なし"
        fe.description(safe_summary)
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

        logger.info(f"処理完了: {ai_title[:15]}...")

    fg.rss_file('rss.xml')
    logger.info("rss.xml 生成完了")

if __name__ == "__main__":
    main()