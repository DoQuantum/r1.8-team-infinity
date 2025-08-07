import json
import csv
import pandas as pd
# https://docs.python.org/3/library/urllib.request.html#module-urllib.request
# This library will be used to fetch the API.
import urllib.request
from dotenv import load_dotenv
import os

from newspaper import Article
from newspaper import Config
import time

# Call API and write results to CSV
def getArticles():
       load_dotenv()
       API_KEY = os.getenv("API_KEY")
       # Query parameters are adjusted in this url
       url = f"https://gnews.io/api/v4/search?q=Google&lang=en&max=10&from=2023-07-18T21:32:58.500Z&to=2024-07-18T21:32:58.500Z&apikey={API_KEY}"
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
       # Set user agent
       config = Config()
       config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'


       # Get dataframe from articles.csv
       df = pd.read_csv("articles.csv")
       for url in df['url']:
              # Print URL and article content
              print("URL:", url)
              article = Article(url, config=config)
              time.sleep(2) # Pause to avoid triggering rate limits
              article.download()
              article.parse()
              print("Content:", article.text)
#getArticles()
getArticleContent()