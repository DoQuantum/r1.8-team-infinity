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
       url = f"https://gnews.io/api/v4/search?q=Google&lang=en&max=10&from=2022-07-18T21:32:58.500Z&to=2025-07-18T21:32:58.500Z&apikey={API_KEY}"
       # Header row
       df = pd.DataFrame({'title': [], 'url': []})

       with urllib.request.urlopen(url) as response:
              data = json.loads(response.read().decode("utf-8"))
              articles = data["articles"]
              for i in range(len(articles)):
                      # Get new row data
                     newRow = {'title': articles[i]["title"],
                      'url': articles[i]["url"]}
                     df.loc[len(df)] = newRow
        # Write titles and URLS to csv
       df.to_csv('articles.csv', index=False)

# Get content from articles.csv
def getArticleContent():

       # Get dataframe from articles.csv
       df = pd.read_csv("articles.csv")
       for url in df['url']:
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
              article = Article(url, config=config)
              time.sleep(2) # Pause to avoid triggering rate limits
              article.download()
              article.parse()
              print("Content:", article.text)

getArticles()
getArticleContent()