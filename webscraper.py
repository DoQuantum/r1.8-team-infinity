import requests
from bs4 import BeautifulSoup
import pandas as pd
from newspaper import Article
from datetime import datetime

start_date = "2020-01-01"
end_date = "2024-12-31"

def scrape_google_news(keyword, start=start_date, end=end_date):
    """Fetch articles from Google News RSS within the date range."""
    query = f"{keyword} after:{start} before:{end}"
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")

    articles = []
    for item in soup.find_all("item"):
        title = item.title.text
        link = item.link.text
        pub_date = item.pubDate.text
        articles.append({"Title": title, "Link": link, "Date": pub_date})
    return articles


def extract_full_text(articles):
    """Download and parse full article text using newspaper3k."""
    for art in articles:
        try:
            article = Article(art["Url"])
            article.download()
            article.parse()
            text = article.text.strip().replace("\n", " ")
        except Exception:
            text = "Unable to extract text"
        art["Content"] = text
    return articles


def export_to_csv(articles, filename="articles.csv"):
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Exported {len(df)} articles to {filename}")


if __name__ == "__main__":
    topic = input("Enter a topic to search for: ")
    print(f"Searching for '{topic}' articles from 2020–2024...")
    results = scrape_google_news(topic)
    results_with_text = extract_full_text(results)
    export_to_csv(results_with_text)
