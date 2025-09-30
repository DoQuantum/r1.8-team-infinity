import json
import pandas as pd
import csv
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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import praw
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Output CSV
target_csv = 'readingArticles.csv'

# Always start fresh
df_target = pd.DataFrame(columns=['title', 'url', 'content'])
df_target.to_csv(target_csv, index=False)

# Get working proxies
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

# Fetch articles from GNews API
def getArticles():
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    url = f"https://gnews.io/api/v4/search?q=Stock&lang=en&max=10&apikey={API_KEY}"

    df_articles = pd.DataFrame(columns=['title', 'url', 'content'])
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        articles = data["articles"]
        for i in range(len(articles)):
            df_articles.loc[len(df_articles)] = {'title': articles[i]['title'], 'url': articles[i]['url'], 'content': None}
        # for art in data.get("articles", []):
        #     df_articles.loc[len(df_articles)] = {'title': art["title"], 'url': art["url"], 'content': None}

    # Save fresh CSV
    df_articles.to_csv(target_csv, index=False)
    return df_articles

# Helper: fetch HTML using random proxy
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

# Fetch content for the articles and save directly
def getArticleContent(df_articles):
    ua = UserAgent()
    proxies = get_proxies()
    authors, dates, contents, keywords, summaries = [], [], [], [], []

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
                dates.append(article.publish_date)
                contents.append(article.text.replace('\n',''))
                keywords.append(article.keywords)
                summaries.append(article.summary)
                print("Content length:", len(article.text))
            except:
                authors.append(None)
                dates.append(None)
                contents.append(None)
                keywords.append(None)
                summaries.append(None)
        else:
            authors.append(None)
            dates.append(None)
            contents.append(None)
            keywords.append(None)
            summaries.append(None)
        time.sleep(random.uniform(2, 5))

    df_articles['authors'] = authors
    df_articles['publish_date'] = dates
    df_articles['content'] = contents
    df_articles['keywords'] = keywords
    df_articles['summary'] = summaries
    df_articles.to_csv(target_csv, index=False, quoting=csv.QUOTE_ALL)

def getSentiments(df_articles):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    txt = []
    for i in range(df_articles.shape[0]):
        txt.append(df_articles.loc[i, 'title'] + " " + df_articles.loc[i,'content'])
    inputs = tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)

    sentiment_labels = [-1, 0, 1]
    sentiment = []
    predicted_indices = torch.argmax(probabilities, dim=1)
    print(predicted_indices)
    for i, idx in enumerate(predicted_indices):
        sentiment.append(sentiment_labels[idx])
        print(f"Article {i+1}: {sentiment_labels[idx]}  ->  {txt[i][:80]}...")
    df_articles['sentiment'] = sentiment
    df_articles.to_csv(target_csv, index=False, quoting=csv.QUOTE_ALL)
# Run workflow
articles_df = getArticles()
getArticleContent(articles_df)
<<<<<<< HEAD
print(f"Saved {len(articles_df)} articles to {target_csv}"
)

#Accuracy notes:
# Find a labeled dataset of financial news data (preferrably one that uses finbert)
# from sklearn.metrics import accuracy_score, precision_score
# add a column for numbers 
# refer to socials branch for code 
=======
print(f"Saved {len(articles_df)} articles to {target_csv}")

# temp for getSentiments
# articles_df = pd.read_csv('readingArticles.csv')

getSentiments(articles_df)

>>>>>>> b5178223c05be37b216d1c10d30b4d1c86a5fe05
