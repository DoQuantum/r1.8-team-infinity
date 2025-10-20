import praw
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt
import time
import re

def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_probs(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1)

def fetch_reddit_posts(subreddit_name, years_back=5, max_posts=1000):
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("USER_AGENT")
    )

    subreddit = reddit.subreddit(subreddit_name)
    cutoff_timestamp = int((dt.datetime.now() - dt.timedelta(days=years_back*365)).timestamp())

    results = []
    print(f"Fetching posts from r/{subreddit_name} since {dt.datetime.fromtimestamp(cutoff_timestamp)}...")

    for submission in subreddit.new(limit=None):
        if submission.created_utc < cutoff_timestamp:
            break
        results.append({
            "title": submission.title,
            "body": submission.selftext,
            "created_utc": submission.created_utc
        })
        if len(results) >= max_posts:
            break

    print(f"Collected {len(results)} posts")
    return results

def main():
    start_total = time.time()

    # ---- Fetch Reddit posts ----
    start_reddit = time.time()
    results = fetch_reddit_posts("wallstreetbets", years_back=5, max_posts=2000)
    txt = [clean_text(item["title"] + " " + item["body"]) for item in results]

    # 👇 Confirm date range of fetched posts
    if results:
        oldest = min(results, key=lambda x: x["created_utc"])
        newest = max(results, key=lambda x: x["created_utc"])
        print(f"\nOldest post: {dt.datetime.fromtimestamp(oldest['created_utc'])}")
        print(f"Newest post: {dt.datetime.fromtimestamp(newest['created_utc'])}")

        print("\nExample oldest post:")
        print(f"Title: {oldest['title']}")
        print(f"Date: {dt.datetime.fromtimestamp(oldest['created_utc'])}")

    # ---- Load FinBERT ----
    fin_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fin_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    sentiment_labels = ["negative", "neutral", "positive"]
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}

    # ---- Predict sentiment ----
    batch_size = 50
    for i in range(0, len(txt), batch_size):
        batch_texts = txt[i:i+batch_size]
        probs = get_probs(batch_texts, fin_tokenizer, fin_model)
        pred_indices = torch.argmax(probs, dim=1)
        for j, idx in enumerate(pred_indices):
            label = sentiment_labels[idx]
            num_value = sentiment_map[label]
            print(f"Post {i+j+1}: {label} ({num_value}) -> {batch_texts[j][:80]}...")

    end_reddit = time.time()
    print(f"\nReddit inference took {end_reddit - start_reddit:.2f} seconds")

    end_total = time.time()
    print(f"Total runtime: {end_total - start_total:.2f} seconds")

if __name__ == "__main__":
    main()
