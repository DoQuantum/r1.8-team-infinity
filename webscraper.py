import requests
from bs4 import BeautifulSoup
import pandas as pd
from newspaper import Article
from datetime import datetime as time

import base64
import urllib.request
from dotenv import load_dotenv
import os
import requests
from newspaper import Article
from fake_useragent import UserAgent
import time
import random
from lxml.html import fromstring
import urllib3
from urllib.parse import quote
import re
from urllib.parse import unquote
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

target_stock = "AMZN" # Manually change, for now
target_csv = f'readingArticles_{target_stock}.csv'
target_from = '2025-01-01'
target_to = '2025-02-01'

def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = []
    for i in parser.xpath('//tbody/tr')[:100]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            proxy = ":".join([i.xpath('.//td[1]/text()')[0],
                              i.xpath('.//td[2]/text()')[0]])
            proxies.append(proxy)

    print(f"Scraped {len(proxies)} proxies, now testing them...")
    alive_proxies = []
    test_url = "https://httpbin.org/ip"
    for proxy in proxies:
        try:
            r = requests.get(test_url,
                             proxies={"http": f"http://{proxy}", "https": f"http://{proxy}"},
                             timeout=5)
            if r.status_code == 200:
                alive_proxies.append(proxy)
        except:
            continue
    print(f"Final working proxies: {len(alive_proxies)}")
    return alive_proxies

def fetch_with_proxies(url, headers, proxies_list, max_retries=3):
    for _ in range(max_retries):
        proxy = random.choice(proxies_list) if proxies_list else None
        PROXIES = {"http": f"http://{proxy}", "https": f"http://{proxy}"} if proxy else None
        try:
            resp = requests.get(url, headers=headers, proxies=PROXIES, timeout=15, verify=False)
            if resp.status_code == 200:
                return resp.text
        except:
            continue
    return None

# def resolve_google_news_url(google_news_url):
#     try:
#         # Follow redirects; Google News usually returns a 302 redirect to the real article
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(google_news_url, headers=headers, allow_redirects=True, timeout=10)
#         print(response.url)
#         return response.url  # The final destination URL (real source)
#     except Exception as e:
#         print(f"Error resolving {google_news_url}: {e}")
#         return None

def scrape_google_news(keyword, start=target_from, end=target_to):
    """Fetch articles from Google News RSS within the date range."""
    query = f"{keyword} after:{start} before:{end}"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, allow_redirects=True)
    soup = BeautifulSoup(response.content, "xml")

    articles = pd.DataFrame(columns=['title','url','date'])
    for item in soup.find_all("item",limit=3):
        title = item.title.text
        link = item.link.text
        pub_date = item.pubDate.text
        # articles.append({"title": title, "url": link, "date": pub_date})
        articles.loc[len(articles)] = {'title': title, 'url': extract_real_url(link), 'date': pub_date}
    return articles

def getArticleContent(df_articles):
    ua = UserAgent()
    proxies = get_proxies()
    authors, contents, keywords, summaries, to_keep = [], [], [], [], []
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
                try:
                    article.nlp()  # enable summary and keywords
                except:
                    pass
                authors.append(article.authors)
                contents.append(article.text.replace('\n',''))
                keywords.append(article.keywords)
                summaries.append(article.summary)
                print("Content length:", len(article.text))
                if not 'Please disable your ad-blocker' in article.text:    # Keep article if there are no errors
                    to_keep.append(index)
            except:
                authors.append(None)
                contents.append(None)
                keywords.append(None)
                summaries.append(None)
        else:
            authors.append(None)
            contents.append(None)
            keywords.append(None)
            summaries.append(None)
        time.sleep(random.uniform(2, 5))
        index += 1

    df_articles['authors'] = authors
    df_articles['content'] = contents
    df_articles['keywords'] = keywords
    df_articles['summary'] = summaries
    df_articles = df_articles.iloc[to_keep]     # Drop all rows with no content
    return df_articles


def export_to_csv(articles, filename=target_csv):
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Exported {len(df)} articles to {filename}")


if __name__ == "__main__":
    topic = input("Enter a topic to search for: ")
    print(f"Searching for '{topic}' articles from 2020–2024...")
    results = scrape_google_news(topic)
    results_with_text = getArticleContent(results)
    export_to_csv(results_with_text)