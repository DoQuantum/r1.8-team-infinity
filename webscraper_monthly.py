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

TEST_URL = "https://httpbin.org/ip"

run_all = True
target_stocks = list(pd.read_csv("targetStocks.csv")['stock'])
print(target_stocks)
target_index = 1
num_articles = 50  # Increased to grab more articles per month
target_csv = ''
target_from_year = 2020
target_to_year = 2023

def scrape_proxies():
    url = "https://free-proxy-list.net/"
    response = requests.get(url)
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
        async with session.get(TEST_URL, proxy=proxy_url, timeout=3) as resp:
            if resp.status == 200:
                return proxy
    except:
        return None


async def check_proxies(proxies):
    timeout = aiohttp.ClientTimeout(total=3)
    connector = aiohttp.TCPConnector(limit=200)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [test_proxy(session, proxy) for proxy in proxies]
        results = await asyncio.gather(*tasks)
    return [p for p in results if p]


def get_proxies():
    proxies = scrape_proxies()
    print(f"Scraped {len(proxies)} proxies, testing...")
    alive = asyncio.run(check_proxies(proxies))
    print(f"Working proxies: {len(alive)}")
    return alive


async def fetch_with_proxies_async(url, headers, proxies_list, max_retries=3):
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(max_retries):
            proxy = random.choice(proxies_list) if proxies_list else None
            proxy_url = f"http://{proxy}" if proxy else None
            try:
                async with session.get(
                    url,
                    headers=headers,
                    proxy=proxy_url,
                    ssl=False
                ) as resp:
                    if resp.status == 200:
                        return await resp.text()
            except:
                continue
    return None


def fetch_with_proxies(url, headers, proxies_list, max_retries=3):
    return asyncio.run(fetch_with_proxies_async(url, headers, proxies_list, max_retries))


def resolve_google_news_url(google_news_url):
    interval_time = 1
    try:
        decoded_url = gnewsdecoder(google_news_url, interval=interval_time)
        if decoded_url.get("status"):
            return decoded_url["decoded_url"]
        else:
            print("Error:", decoded_url["message"])
            return google_news_url
    except Exception as e:
        print(f"Error occurred: {e}")
        return google_news_url


def scrape_google_news_month(keyword, year, month):
    """Fetch as many articles as possible from Google News RSS for an entire month."""
    days_in_month = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"
    # End is the first day of the next month
    if month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{month + 1:02d}-01"

    query = f'"{keyword}" after:{start} before:{end}'
    url = f'https://news.google.com/rss/search?q={query.replace(" ", "+")}&tbs=qdr:m'
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, allow_redirects=True)
    soup = BeautifulSoup(response.content, "xml")

    articles = pd.DataFrame(columns=['title', 'url', 'date'])
    for item in soup.find_all("item", limit=num_articles):
        real_url = resolve_google_news_url(item.link.text)
        if "obituaries" not in real_url and "obituary" not in real_url:
            print("Decoded URL:", real_url)
            title = item.title.text
            pub_date = item.pubDate.text
            articles.loc[len(articles)] = {
                "title": title,
                "url": real_url,
                "date": pub_date
            }
    return articles


def normalize_date(date_str):
    """Parse a pubDate string and return a normalized YYYY-MM-DD string."""
    try:
        return pd.to_datetime(date_str, utc=True).strftime("%Y-%m-%d")
    except Exception:
        return date_str


def getArticleContent(df_articles):
    ua = UserAgent()
    proxies = get_proxies()
    authors, contents, to_keep = [], [], []
    index = 0
    for url in df_articles['url']:
        print("\nURL:", url)
        headers = {"User-Agent": ua.random}
        html = fetch_with_proxies(url, headers, proxies)
        if html:
            try:
                article = Article(url)
                article.set_html(html)
                article.parse()
                text = article.text.replace('\n', '')
                if text and text not in contents:
                    authors.append(article.authors)
                    contents.append(text)
                    to_keep.append(index)
                    print("Content length:", len(text))
                else:
                    authors.append(None)
                    contents.append(None)
            except:
                authors.append(None)
                contents.append(None)
        else:
            authors.append(None)
            contents.append(None)
        time.sleep(random.uniform(2, 4))
        index += 1

    df_articles['authors'] = authors
    df_articles['content'] = contents
    df_articles = df_articles.iloc[to_keep].reset_index(drop=True)
    return df_articles


def getSentiments(df_articles):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    txt = []
    for index, row in df_articles.iterrows():
        snippet = row['title'] + " " + ". ".join(row['content'].split(".")[:2])
        txt.append(snippet)
    txt = [t for t in txt if t.strip()]

    if len(txt) == 0:
        return None

    try:
        inputs = tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
    except Exception as e:
        print(f"Tokenizer error: {e}")
        for t in txt:
            print(t)
        return None

    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)

    print(model.config.id2label)
    sentiment_labels = [1, -1, 0]
    predicted_indices = torch.argmax(probabilities, dim=1)
    print(predicted_indices)

    sentiment = []
    for i, idx in enumerate(predicted_indices):
        sentiment.append(sentiment_labels[idx])
        print(f"Article {i+1}: {sentiment_labels[idx]}  ->  {txt[i][:80]}...")

    df_articles['sentiment'] = sentiment
    return df_articles


def evaluate_accuracy():
    print("\nEvaluating accuracy using sentences_allagree.csv")
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

    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (macro): {prec:.3f}")


def export_to_csv(articles, filename):
    df = pd.DataFrame(articles)
    file_path = Path(filename)
    if file_path.is_file():
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Exported {len(df)} articles to {filename}")


def run_company(index):
    global target_csv
    start_article_get = time.time()
    topic = target_stocks[index]
    target_csv = f'sentiment-data/articles_{target_stocks[index]}.csv'

    temp_year = 2020
    print(index)
    if index == 0:
        temp_year = 2020

    for year in range(temp_year, target_to_year + 1):
        temp_month = 1
        if year == temp_year and index == 0:
            temp_month = 11

        for month in range(temp_month, 13):
            try:
                print(f"\n=== Fetching articles for '{topic}' — {year}-{month:02d} ===")

                # --- Step 1: Scrape all articles for the month ---
                month_articles = scrape_google_news_month(topic, year, month)

                if month_articles is None or month_articles.empty:
                    print("No articles found for this month.")
                    continue

                # --- Step 2: Fetch full article content ---
                month_articles = getArticleContent(month_articles)

                if month_articles is None or month_articles.empty:
                    print("No content retrieved for this month.")
                    continue

                # --- Step 3: Run sentiment analysis ---
                month_articles = getSentiments(month_articles)

                if month_articles is None or 'sentiment' not in month_articles.columns:
                    print("Sentiment analysis failed for this month.")
                    continue

                # --- Step 4: Normalize dates to YYYY-MM-DD ---
                month_articles['date'] = month_articles['date'].apply(normalize_date)

                # --- Step 5: Sort by date, then group by date to compute avg sentiment ---
                month_articles = month_articles.sort_values('date').reset_index(drop=True)
                daily_sentiment = (
                    month_articles.groupby('date')['sentiment']
                    .mean()
                    .reset_index()
                    .rename(columns={'sentiment': 'avg_sentiment'})
                )

                print(daily_sentiment)

                # --- Step 6: Append daily results to CSV ---
                export_to_csv(daily_sentiment, target_csv)

                end_article_get = time.time()
                print(f"Month processed in {end_article_get - start_article_get:.2f}s")
            except:
                time.sleep(random.uniform(3, 5))
                run_company(index,year,month)


if __name__ == "__main__":
    if run_all:
        for i in range(0,30):
            run_company(i)
    else:
        run_company(target_index)