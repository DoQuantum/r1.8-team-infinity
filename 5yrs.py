import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import requests
import datetime as dt
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd  

# --- Load finetuned model ---
model_name_or_path = "./finbert_finetuned"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

reverse_map = {0: -1, 1: 0, 2: 1}
MAX_RETRIES = 5
SUBREDDIT = "wallstreetbets"
MAX_POSTS_PER_QUARTER = 50

def get_quarters_2020_2025():
    quarters = []
    for year in range(2020, 2026):
        quarters.append((int(dt.datetime(year, 1, 1).timestamp()), int(dt.datetime(year, 3, 31, 23, 59, 59).timestamp())))
        quarters.append((int(dt.datetime(year, 4, 1).timestamp()), int(dt.datetime(year, 6, 30, 23, 59, 59).timestamp())))
        quarters.append((int(dt.datetime(year, 7, 1).timestamp()), int(dt.datetime(year, 9, 30, 23, 59, 59).timestamp())))
        quarters.append((int(dt.datetime(year, 10, 1).timestamp()), int(dt.datetime(year, 12, 31, 23, 59, 59).timestamp())))
    return quarters

def fetch_posts_by_date_range(company, subreddit, start_ts, end_ts, max_posts=500):
    all_posts = []
    last_created_utc = end_ts
    company = company.upper()
    encoded_query = f"{company}%20OR%20%24{company}"

    print(f"Fetching posts for {company} between {dt.datetime.fromtimestamp(start_ts)} and {dt.datetime.fromtimestamp(end_ts)}")

    while len(all_posts) < max_posts:
        size = min(100, max_posts - len(all_posts))
        url = (
            f"https://api.pullpush.io/reddit/search/submission/"
            f"?subreddit={subreddit}&after={start_ts}&before={last_created_utc}"
            f"&size={size}&q={encoded_query}&sort=desc"
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    break
                print(f"Attempt {attempt+1} failed with status {response.status_code}")
            except requests.RequestException as e:
                print(f"Attempt {attempt+1} exception: {e}")
            time.sleep(5 * (attempt + 1))  # exponential backoff
        else:
            print(f"Failed to fetch posts for {company} after {MAX_RETRIES} attempts")
            break

        data = response.json().get("data", [])
        if not data:
            break

        for post in data:
            title = post.get("title", "").strip()
            body = post.get("selftext", "").strip()
            if not title or title.lower() in ("[deleted]", "[removed]") or body.lower() in ("[deleted]", "[removed]"):
                continue
            all_posts.append({
                "company": company,
                "title": title,
                "body": body,
                "created_utc": post.get("created_utc", 0),
                "url": f"https://www.reddit.com{post.get('permalink', '')}"
            })

        last_created_utc = data[-1]["created_utc"] - 1
        print(f"Collected {len(all_posts)} valid posts so far for {company}...")
        time.sleep(1)

    return all_posts

def predict_sentiment(posts):
    for post in posts:
        text = post["title"] + " " + post.get("body", "")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()
        post["sentiment"] = reverse_map[pred_label]
        post["created_date"] = dt.datetime.fromtimestamp(post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S")
    return posts

if __name__ == "__main__":
    snp_csv = "tickers_unique.csv"
    snp_df = pd.read_csv(snp_csv)

    all_results = []

    quarters = get_quarters_2020_2025()

    for company_name in snp_df["company"].unique():
        print(f"\n=== Processing {company_name} ===")
        for start_ts, end_ts in quarters:
            posts = fetch_posts_by_date_range(company_name, SUBREDDIT, start_ts, end_ts, max_posts=MAX_POSTS_PER_QUARTER)
            if not posts:
                print(f"No posts fetched for {company_name} this quarter")
                continue
            scored_posts = predict_sentiment(posts)
            all_results.extend(scored_posts)

    if all_results:
        df = pd.DataFrame(all_results)[["company", "created_date", "title", "body", "sentiment", "url"]]
        output_file = "all_companies_2020_2025_sentiments.csv"
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"\nSaved total {len(df)} posts to {output_file}")
    else:
        print("No posts fetched for any company.")
