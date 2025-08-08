import json
import csv
import pandas as pd
# https://docs.python.org/3/library/urllib.request.html#module-urllib.request
# This library will be used to fetch the API.
import urllib.request
from dotenv import load_dotenv
import os

import requests
from newspaper import Article
from newspaper import Config
from fake_useragent import UserAgent
import time
import random

old_df = pd.read_csv('articles.csv')

# Manually change this to decide between overwriting the file or appending. True for append, and False for overwrite
append = True

# Get list of proxies to rotate while scraping articles
from lxml.html import fromstring
def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = []
    for i in parser.xpath('//tbody/tr')[:100]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            #Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0],
            i.xpath('.//td[2]/text()')[0]])              
            #print(proxy)
            proxies.append(proxy)
    return proxies
proxies = get_proxies()

# Call API and write results to CSV
def getArticles():
       load_dotenv()
       API_KEY = os.getenv("API_KEY")
       # Query parameters are adjusted in this url
       url = f"https://gnews.io/api/v4/search?q=Google&lang=en&max=10&from=2022-06-28T21:32:58.500Z&to=2025-06-28T21:32:58.500Z&apikey={API_KEY}"
       # Header row
       new_df = pd.DataFrame({'title': [], 'url': []})

       with urllib.request.urlopen(url) as response:
              data = json.loads(response.read().decode("utf-8"))
              articles = data["articles"]
              for i in range(len(articles)):
                      # Get new row data
                     newRow = {'title': articles[i]["title"],
                      'url': articles[i]["url"]}
                     new_df.loc[len(new_df)] = newRow
       # Write titles and URLS to csv
       if append:
              combined_df = pd.concat([old_df, new_df], ignore_index=True)
              combined_df.to_csv('articles.csv', index=False)
       else:
              new_df.to_csv('articles.csv', index=False)


# Get content from articles.csv
def getArticleContent():
       contents = []

       # Get dataframe from articles.csv
       df_tail = pd.read_csv("articles.csv")

       # Only fetch content for the newest 10 articles (useful when doing append mode)
       df_tail = df_tail.tail(10).reset_index(drop=True)

       for url in df_tail['url']:
              print()
              
              # Set a random user agent
              config = Config()
              ua = UserAgent()
              config.browser_user_agent = ua.random

              # Set proxies
              # proxy = random.choice(proxies) # to use random proxy
              proxy = "38.147.98.190:8080"
              PROXIES = {
                     'http': f"http://{proxy}",
                     'https': f"http://{proxy}"
              }
              print("Using proxy: ", proxy)
              config.proxies = PROXIES
              # Set timeout
              config.request_timeout = 10

              # Print URL and article content
              print("URL:", url)
              article = Article(url, config=config, verify=False)
              time.sleep(2) # Pause to avoid triggering rate limits
              article.download()

              # IMPORTANT!!! Need to figure out why we encounter errors with connection (Example of error below)
              '''newspaper.article.ArticleException: Article `download()` failed with HTTPSConnectionPool(host='www.androidheadlines.com', port=443): 
              Max retries exceeded with url: /2025/06/google-photos-editor-is-getting-a-major-redesign-soon-heres-the-first-look.html 
              (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in 
              certificate chain (_ssl.c:1000)'))) on URL https://www.androidheadlines.com/2025/06/google-photos-editor-is-getting-a-major-redesign-soon-heres-the-first-look.html
              '''
              try:
                     article.parse()
                     contents.append(article.text)
              except:
                    print("Article download() failed")
                    contents.append(None)
              print("Content:", article.text)

       df_tail['content'] = contents

       # Append or update the CSV
       df = pd.read_csv("articles.csv")
       df.loc[df.tail(10).index, 'content'] = df_tail['content']
       df.to_csv('articles.csv', index=False)

getArticles()
getArticleContent()