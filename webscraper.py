# import modules
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
from newspaper import Article
from datetime import datetime as time
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
from googlenewsdecoder import gnewsdecoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from sklearn.metrics import accuracy_score, precision_score

run_all = True
target_stocks = list(pd.read_csv("targetStocks.csv")['stock'])
print(target_stocks)
target_index = 1
num_articles = 10
target_csv = ''
target_from_year = 2020
target_to_year = 2023

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

def resolve_google_news_url(google_news_url):
    interval_time = 1  # interval is optional, default is None
    try:
        decoded_url = gnewsdecoder(google_news_url, interval=interval_time)

        if decoded_url.get("status"):
            print("Decoded URL:", decoded_url["decoded_url"])
            return decoded_url["decoded_url"]
        else:
            print("Error:", decoded_url["message"])
            return google_news_url
    except Exception as e:
        print(f"Error occurred: {e}")
        return google_news_url

def scrape_google_news(keyword, start, end):

    """Fetch articles from Google News RSS within the date range."""
    query = f"{keyword} after:{start} before:{end}"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&tbs=qdr:d"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, allow_redirects=True)
    soup = BeautifulSoup(response.content, "xml")

    articles = pd.DataFrame(columns=['title','url','date'])
    for item in soup.find_all("item", limit=num_articles):
        real_url = resolve_google_news_url(item.link.text)
        if not "aboutamazon" in real_url:
            title = item.title.text
            pub_date = item.pubDate.text

            articles.loc[len(articles)] = {
                "title": title,
                "url": real_url,
                "date": pub_date
            }
    return articles

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

                # only append if non-empty and not already in list
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
        time.sleep(random.uniform(2, 5))
        index += 1

    df_articles['authors'] = authors
    df_articles['content'] = contents
    df_articles = df_articles.iloc[to_keep]     # Drop all rows with no content
    return df_articles

def getSentiments(df_articles):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    txt = []
    for index, row in df_articles.iterrows():
        txt.append(row['title'] + " " + row['content'])
    inputs = tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)

    print(model.config.id2label)
    sentiment_labels = [1, -1, 0]
    sentiment = []
    predicted_indices = torch.argmax(probabilities, dim=1)
    print(predicted_indices)
    for i, idx in enumerate(predicted_indices):
        sentiment.append(sentiment_labels[idx])
        print(f"Article {i+1}: {sentiment_labels[idx]}  ->  {txt[i][:80]}...")
    df_articles['sentiment'] = sentiment
    df_articles.to_csv(target_csv, index=False, quoting=csv.QUOTE_ALL)

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
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Exported {len(df)} articles to {filename}")

def run_company(index):
    global target_csv
    # start_total = time.time()
    start_article_get = time.time()
    topic = target_stocks[index]
    target_csv = f'sentiment-data/articles_{target_stocks[index]}.csv'
    results = pd.DataFrame()
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    for year in range (target_from_year,target_to_year+1):
        for month in range (1,13):
            for day in range(1,days[month - 1]+1):
                start = f"{year}-{month}-{day}"
                # Edge cases for end date
                if day == days[month - 1]:
                    if month == 12:
                        end = f"{year+1}-1-1"
                    else:
                        end = f"{year}-{month+1}-1"
                else:
                    end = f"{year}-{month}-{day}"
                # Get URLs, content, and sentiments for each article
                print(f"Searching for '{topic}' articles from {start} to {end}...")
                results_day = scrape_google_news(topic,start,end)
                results_day = getArticleContent(results_day)
                getSentiments(results_day)
                # TODO
                # for each row, add to total and divide by # of rows
                # add row (date,avg sentiment) to the results dataframe
                avg_sentiment = 0


                end_article_get = time.time()
                print(f"Article collecting took {end_article_get - start_article_get:.2f} seconds")
                #start_acc_eval = time.time()
                #evaluate_accuracy()
                #end_acc_eval = time.time()
                #print(f"Accuracy evaluation took {end_acc_eval - start_acc_eval:.2f} seconds")
                # end_total = time.time()
                # print(f"\nTotal program runtime: {end_total - start_total:.2f} seconds")
    export_to_csv(results, target_csv)
if __name__ == "__main__":
    if run_all:
        for i in range(30):
            run_company(i)
    else:
        run_company(target_index)