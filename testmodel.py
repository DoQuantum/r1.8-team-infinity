import requests
import datetime as dt
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd  # <-- added

# --- Load finetuned model ---
model_name_or_path = "./finbert_finetuned"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Sentiment mapping back to original labels ---
reverse_map = {0: -1, 1: 0, 2: 1}

# --- Reddit fetching function ---
def fetch_old_posts(company, subreddit, year=2020, max_posts=500):
    after = int(dt.datetime(year, 1, 1).timestamp())
    before = int(dt.datetime(year + 1, 1, 1).timestamp())
    all_posts = []
    last_created_utc = before

    print(f"Fetching posts from r/{subreddit} in {year}...")

    while len(all_posts) < max_posts:
        size = min(100, max_posts - len(all_posts))
        url = (
            f"https://api.pullpush.io/reddit/search/submission/"
            f"?subreddit={subreddit}&after={after}&before={last_created_utc}"
            f"&size={size}&q={company}&sort=desc"
        )
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Request failed with status {response.status_code}")
            break
        data = response.json().get("data", [])
        if not data:
            break

        for post in data:
            title = post.get("title", "").strip()
            body = post.get("selftext", "").strip()
            if not title or not body:
                continue
            if title.lower() in ("[deleted]", "[removed]") or body.lower() in ("[deleted]", "[removed]"):
                continue

            all_posts.append({
                "title": title,
                "body": body,
                "created_utc": post.get("created_utc", 0),
                "url": f"https://www.reddit.com{post.get('permalink', '')}"
            })

        last_created_utc = data[-1]["created_utc"]
        print(f"Collected {len(all_posts)} valid posts so far...")
        time.sleep(1)

    return all_posts


# --- Sentiment prediction ---
def predict_sentiment(posts):
    predictions = []
    for post in posts:
        text = post["title"] + " " + post["body"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()
        post["sentiment"] = reverse_map[pred_label]
        predictions.append(post)
    return predictions


# --- Example Run ---
if __name__ == "__main__":
    posts_2020 = fetch_old_posts("Amazon", "wallstreetbets", year=2020, max_posts=500)
    scored_posts = predict_sentiment(posts_2020)

    # Convert Unix timestamps to readable dates
    for post in scored_posts:
        post["created_date"] = dt.datetime.fromtimestamp(post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S")

    # Save results to CSV
    df = pd.DataFrame(scored_posts)
    df = df[["created_date", "title", "body", "sentiment", "url"]]  # optional: select relevant columns
    output_path = "reddit_sentiments.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ Saved all {len(df)} posts with sentiment scores to {output_path}")

    # Print sample output
    for i, post in enumerate(scored_posts[:5], 1):
        print(f"\n[{i}] {post['created_date']}")
        print(f"Title: {post['title']}")
        print(f"Sentiment: {post['sentiment']} (0=neutral, 1=positive, -1=negative)")
        print(f"URL: {post['url']}")

