# import modules
import requests
from bs4 import BeautifulSoup
import pandas as pd
from newspaper import Article
import time
import random
from lxml.html import fromstring
import urllib3
from urllib.parse import quote
from urllib.parse import unquote
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from googlenewsdecoder import gnewsdecoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import asyncio
import aiohttp
import calendar
from fake_useragent import UserAgent
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_URL = "https://httpbin.org/ip"

run_all = True
target_stocks = list(pd.read_csv("targetStocks.csv")['stock'])
log.info(f"Target stocks: {target_stocks}")
target_index = 1
num_articles = 50
target_from_year = 2020
target_to_year = 2023

# Domains that are paywalled or consistently block scraping — skip them early
BLOCKED_DOMAINS = {
    "wsj.com", "bloomberg.com", "ft.com", "barrons.com",
    "economist.com", "thetimes.co.uk", "nytimes.com","businesswire.com","gulfbusiness.com",
    "reuters.com","pubs.acs.org"
}

# Per-domain minimum delay in seconds to avoid rate limiting
DOMAIN_DELAYS = {
    "reuters.com": 3,
    "cnbc.com": 3,
    "marketwatch.com": 4,
    "seekingalpha.com": 5,
}
DEFAULT_DELAY = 2


def is_blocked_domain(url: str) -> bool:
    """Return True if the URL belongs to a known paywalled/blocked domain."""
    try:
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        return any(blocked in hostname for blocked in BLOCKED_DOMAINS)
    except Exception:
        return False


def domain_delay(url: str) -> float:
    """Return appropriate sleep time for the domain."""
    try:
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        for domain, delay in DOMAIN_DELAYS.items():
            if domain in hostname:
                return delay
    except Exception:
        pass
    return DEFAULT_DELAY


# ── Proxy helpers ──────────────────────────────────────────────────────────────

def scrape_proxies():
    url = "https://free-proxy-list.net/"
    response = requests.get(url, timeout=10)
    parser = fromstring(response.text)
    proxies = []
    for i in parser.xpath("//tbody/tr")[:100]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([
                i.xpath(".//td[1]/text()")[0],
                i.xpath(".//td[2]/text()")[0]
            ])
            proxies.append(proxy)
    return proxies


async def test_proxy(session, proxy):
    try:
        proxy_url = f"http://{proxy}"
        async with session.get(TEST_URL, proxy=proxy_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
            if resp.status == 200:
                return proxy
    except Exception:
        return None


async def check_proxies(proxies):
    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [test_proxy(session, proxy) for proxy in proxies]
        results = await asyncio.gather(*tasks)
    return [p for p in results if p]


def get_proxies():
    try:
        proxies = scrape_proxies()
        log.info(f"Scraped {len(proxies)} proxies, testing...")
        alive = asyncio.run(check_proxies(proxies))
        log.info(f"Working proxies: {len(alive)}")
        return alive
    except Exception as e:
        log.warning(f"Proxy scrape failed: {e}. Continuing without proxies.")
        return []


# ── Fetch helpers ──────────────────────────────────────────────────────────────

async def fetch_with_proxies_async(url, headers, proxies_list, max_retries=3):
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Try with proxies first, then fall back to direct
        candidates = []
        if proxies_list:
            candidates = [random.choice(proxies_list) for _ in range(max_retries - 1)]
        candidates.append(None)  # final fallback: direct request

        for proxy in candidates:
            proxy_url = f"http://{proxy}" if proxy else None
            try:
                async with session.get(url, headers=headers, proxy=proxy_url, ssl=False) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    else:
                        log.debug(f"HTTP {resp.status} for {url} via {'proxy' if proxy else 'direct'}")
            except Exception as e:
                log.debug(f"Fetch error ({type(e).__name__}) for {url}: {e}")
                continue
    return None


def fetch_with_proxies(url, headers, proxies_list, max_retries=3):
    return asyncio.run(fetch_with_proxies_async(url, headers, proxies_list, max_retries))


# ── Google News helpers ────────────────────────────────────────────────────────

def resolve_google_news_url(google_news_url):
    interval_time = 1
    try:
        decoded_url = gnewsdecoder(google_news_url, interval=interval_time)
        if decoded_url.get("status"):
            return decoded_url["decoded_url"]
        else:
            log.warning(f"URL decode error: {decoded_url.get('message')}")
            return google_news_url
    except Exception as e:
        log.warning(f"resolve_google_news_url error: {e}")
        return google_news_url


def scrape_google_news_month(keyword, year, month):
    """Fetch articles from Google News RSS for an entire month."""
    start = f"{year}-{month:02d}-01"
    end = f"{year + 1}-01-01" if month == 12 else f"{year}-{month + 1:02d}-01"

    query = f'"{keyword}" after:{start} before:{end}'
    url = f'https://news.google.com/rss/search?q={query.replace(" ", "+")}&tbs=qdr:m'
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
        response.raise_for_status()
    except Exception as e:
        log.error(f"Google News RSS fetch failed: {e}")
        return pd.DataFrame(columns=['title', 'url', 'date'])

    soup = BeautifulSoup(response.content, "xml")
    articles = pd.DataFrame(columns=['title', 'url', 'date'])

    for item in soup.find_all("item", limit=num_articles):
        try:
            real_url = resolve_google_news_url(item.link.text)
            if "obituar" in real_url.lower():
                continue
            if is_blocked_domain(real_url):
                log.info(f"Skipping paywalled domain: {real_url}")
                continue
            log.info(f"Decoded URL: {real_url}")
            articles.loc[len(articles)] = {
                "title": item.title.text,
                "url": real_url,
                "date": item.pubDate.text
            }
        except Exception as e:
            log.warning(f"Skipping RSS item due to error: {e}")

    return articles


def normalize_date(date_str):
    try:
        return pd.to_datetime(date_str, utc=True).strftime("%Y-%m-%d")
    except Exception:
        return date_str


# ── Article content retrieval ──────────────────────────────────────────────────

def getArticleContent(df_articles):
    ua = UserAgent()
    proxies = get_proxies()
    authors, contents, to_keep = [], [], []

    # URL-based dedup instead of full-text comparison (faster + more accurate)
    seen_urls = set()

    for index, row in df_articles.iterrows():
        url = row['url']
        log.info(f"\nFetching [{index + 1}/{len(df_articles)}]: {url}")

        if url in seen_urls:
            log.info("Duplicate URL, skipping.")
            authors.append(None)
            contents.append(None)
            continue
        seen_urls.add(url)

        headers = {"User-Agent": ua.random}
        html = fetch_with_proxies(url, headers, proxies)

        if html:
            try:
                article = Article(url)
                article.set_html(html)
                article.parse()
                text = article.text.replace('\n', ' ').strip()
                if text:
                    authors.append(article.authors)
                    contents.append(text)
                    to_keep.append(index)
                    log.info(f"Content length: {len(text)} chars")
                else:
                    log.warning(f"Empty content parsed from {url}")
                    authors.append(None)
                    contents.append(None)
            except Exception as e:
                log.warning(f"Article parse error for {url}: {type(e).__name__}: {e}")
                authors.append(None)
                contents.append(None)
        else:
            log.warning(f"No HTML retrieved for {url}")
            authors.append(None)
            contents.append(None)

        delay = domain_delay(url) + random.uniform(0, 2)
        log.debug(f"Sleeping {delay:.1f}s")
        time.sleep(delay)

    df_articles = df_articles.copy()
    df_articles['authors'] = authors
    df_articles['content'] = contents
    df_articles = df_articles.iloc[to_keep].reset_index(drop=True)
    return df_articles


# ── Sentiment analysis ─────────────────────────────────────────────────────────

def getSentiments(df_articles):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    txt = []
    for _, row in df_articles.iterrows():
        snippet = row['title'] + " " + ". ".join(str(row['content']).split(".")[:2])
        txt.append(snippet.strip())

    txt = [t for t in txt if t]

    if not txt:
        log.warning("No text available for sentiment analysis.")
        return None

    try:
        inputs = tokenizer(txt, padding=True, truncation=True, max_length=512, return_tensors="pt")
    except Exception as e:
        log.error(f"Tokenizer error: {e}")
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    log.info(f"FinBERT label map: {model.config.id2label}")

    sentiment_labels = [1, -1, 0]  # positive, negative, neutral
    predicted_indices = torch.argmax(probabilities, dim=1)

    sentiment = []
    for i, idx in enumerate(predicted_indices):
        label = sentiment_labels[idx]
        sentiment.append(label)
        log.info(f"Article {i + 1}: sentiment={label}  |  {txt[i][:80]}...")

    df_articles = df_articles.copy()
    df_articles['sentiment'] = sentiment
    return df_articles


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_accuracy():
    log.info("Evaluating accuracy using sentences_allagree.csv")
    df = pd.read_csv("sentences_allagree.csv")

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    texts = df["sentence"].astype(str).tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    predicted_indices = torch.argmax(probs, dim=1)

    sentiment_labels = [1, -1, 0]
    predictions = [sentiment_labels[idx] for idx in predicted_indices]
    ground_truth = df["numerical_sentiment"].tolist()

    acc = accuracy_score(ground_truth, predictions)
    prec = precision_score(ground_truth, predictions, average="macro", zero_division=0)

    log.info(f"Accuracy: {acc:.3f}")
    log.info(f"Precision (macro): {prec:.3f}")


# ── Export ─────────────────────────────────────────────────────────────────────

def export_to_csv(articles, filename):
    df = pd.DataFrame(articles)
    file_path = Path(filename)
    if file_path.is_file():
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=False, encoding="utf-8")
    log.info(f"Exported {len(df)} rows to {filename}")


# ── Main runner ────────────────────────────────────────────────────────────────

def run_company(index, resume_year=None, resume_month=None):
    """
    Process one stock ticker across all months in the target date range.

    Args:
        index:        Index into target_stocks list.
        resume_year:  Optional year to resume from (inclusive).
        resume_month: Optional month to resume from (inclusive, used only when
                      resume_year matches the iteration year).
    """
    start_time = time.time()
    topic = target_stocks[index]
    target_csv = f'sentiment-data/articles_{topic}.csv'
    log.info(f"\n{'='*60}\nProcessing stock [{index}]: {topic}\n{'='*60}")

    # Determine the actual start year/month for this ticker
    default_start_year = target_from_year
    default_start_month = 1

    # Special-case logic preserved from original
    if index == 0:
        default_start_year = 2022
        default_start_month = 11

    start_year = resume_year if resume_year is not None else default_start_year

    for year in range(start_year, target_to_year + 1):
        # Determine start month for this year
        if year == start_year:
            if resume_month is not None and year == resume_year:
                start_month = resume_month
            elif year == default_start_year:
                start_month = default_start_month
            else:
                start_month = 1
        else:
            start_month = 1

        for month in range(start_month, 13):
            log.info(f"\n--- {topic} | {year}-{month:02d} ---")
            try:
                # Step 1: Scrape article URLs for the month
                month_articles = scrape_google_news_month(topic, year, month)
                if month_articles is None or month_articles.empty:
                    log.info("No articles found for this month.")
                    continue

                # Step 2: Fetch full article content
                month_articles = getArticleContent(month_articles)
                if month_articles is None or month_articles.empty:
                    log.info("No content retrieved for this month.")
                    continue

                # Step 3: Sentiment analysis
                month_articles = getSentiments(month_articles)
                if month_articles is None or 'sentiment' not in month_articles.columns:
                    log.warning("Sentiment analysis failed for this month.")
                    continue

                # Step 4: Normalize dates
                month_articles['date'] = month_articles['date'].apply(normalize_date)

                # Step 5: Aggregate to daily average sentiment
                month_articles = month_articles.sort_values('date').reset_index(drop=True)
                daily_sentiment = (
                    month_articles.groupby('date')['sentiment']
                    .mean()
                    .reset_index()
                    .rename(columns={'sentiment': 'avg_sentiment'})
                )
                log.info(f"\n{daily_sentiment}")

                # Step 6: Append to CSV
                export_to_csv(daily_sentiment, target_csv)

                elapsed = time.time() - start_time
                log.info(f"Month processed in {elapsed:.1f}s total elapsed")

            except KeyboardInterrupt:
                log.warning("Interrupted by user.")
                raise
            except Exception as e:
                log.error(f"Unhandled error for {topic} {year}-{month:02d}: {type(e).__name__}: {e}")
                # Back off and move on to the next month rather than infinite recursion
                time.sleep(random.uniform(3, 6))
                continue


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if run_all:
        for i in range(0, len(target_stocks)):
            run_company(i)
    else:
        run_company(target_index)