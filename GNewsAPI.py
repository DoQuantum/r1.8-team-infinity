import json
import csv
import pandas as pd
# https://docs.python.org/3/library/urllib.request.html#module-urllib.request
# This library will be used to fetch the API.
import urllib.request
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
# Query parameters are adjusted in this url
url = f"https://gnews.io/api/v4/search?q=Google&lang=en&max=10&from=2015-07-18T21:32:58.500Z&to=2020-07-18T21:32:58.500Z&apikey={API_KEY}"

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

df.to_csv('articles.csv', index=False)