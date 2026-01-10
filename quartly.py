import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow info logs

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
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Request failed with status {response.status_code}")
            break

        data = response.json().get("data", [])
        print(f"API returned {len(data)} posts this batch")
        if not data:
            break

        for post in data:
            title = post.get("title", "").strip()
            body = post.get("selftext", "").strip()
            if not title:
                continue
            if title.lower() in ("[deleted]", "[removed]") or body.lower() in ("[deleted]", "[removed]"):
                continue
            all_posts.append({
                "title": title,
                "body": body,
                "created_utc": post.get("created_utc", 0),
                "url": f"https://www.reddit.com{post.get('permalink', '')}"
            })

        last_created_utc = data[-1]["created_utc"] - 1
        print(f"Collected {len(all_posts)} valid posts so far...")
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
    return posts

# --- Load S&P500 CSV to get quarters ---
snp_csv = "snp500_quarters.csv"  # replace with your CSV path
snp_df = pd.read_csv(snp_csv, sep="\t")  # adjust sep if needed

def get_quarter_date_range(company, snp_df):
    company = company.upper()
    for _, row in snp_df.iterrows():
        if company in row.values:
            sample_date = dt.datetime.strptime(row[0], "%m/%d/%Y")
            month = sample_date.month
            year = sample_date.year

            if month <= 3:   # Q1
                start_date = dt.datetime(year, 1, 1)
                end_date = dt.datetime(year, 3, 31, 23, 59, 59)
            elif month <= 6: # Q2
                start_date = dt.datetime(year, 4, 1)
                end_date = dt.datetime(year, 6, 30, 23, 59, 59)
            elif month <= 9: # Q3
                start_date = dt.datetime(year, 7, 1)
                end_date = dt.datetime(year, 9, 30, 23, 59, 59)
            else:            # Q4
                start_date = dt.datetime(year, 10, 1)
                end_date = dt.datetime(year, 12, 31, 23, 59, 59)

            return int(start_date.timestamp()), int(end_date.timestamp())
    return None, None

if __name__ == "__main__":
    company_name = "AAPL"  # change to desired company
    subreddit_name = "wallstreetbets"

    start_ts, end_ts = get_quarter_date_range(company_name, snp_df)
    if start_ts and end_ts:
        posts_quarter = fetch_posts_by_date_range(company_name, subreddit_name, start_ts, end_ts, max_posts=50)
        scored_quarter = predict_sentiment(posts_quarter)

        for post in scored_quarter:
            post["created_date"] = dt.datetime.fromtimestamp(post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S")

        df_quarter = pd.DataFrame(scored_quarter)[["created_date", "title", "body", "sentiment", "url"]]
        df_quarter.to_csv(f"{company_name}_quarter_sentiments.csv", index=False, encoding="utf-8")
        print(f"Saved {len(df_quarter)} posts to {company_name}_quarter_sentiments.csv")
    else:
        print(f"No quarter found for {company_name} in the CSV.")
