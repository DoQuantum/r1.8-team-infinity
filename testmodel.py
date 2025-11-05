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

# --- Sentiment mapping back to original labels ---
reverse_map = {0: -1, 1: 0, 2: 1}  # negative, neutral, positive

# --- Clean text ---
def clean_text(text):
    return " ".join(text.split())  # removes newlines and extra spaces

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
            title = clean_text(post.get("title", ""))
            body = clean_text(post.get("selftext", ""))
            if not title and not body:
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
    for post in posts:
        text = post["title"] + " " + post["body"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()
        post["sentiment"] = reverse_map[pred_label]
        post["created_date"] = dt.datetime.fromtimestamp(post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S")
    # Filter out posts that somehow ended up empty
    return [p for p in posts if p["title"] or p["body"]]

# --- Main ---
if __name__ == "__main__":
    company = "Amazon"
    subreddit = "wallstreetbets"
    posts = fetch_old_posts(company, subreddit, year=2020, max_posts=500)
    scored_posts = predict_sentiment(posts)

    df = pd.DataFrame(scored_posts)
    df = df[["created_date", "title", "body", "sentiment", "url"]]
    df.to_csv("reddit_sentiments.csv", index=False, encoding="utf-8")
    print(f"\n✅ Saved {len(df)} posts cleanly to reddit_sentiments.csv")

    # Print sample output
    for i, post in enumerate(scored_posts[:5], 1):
        print(f"\n[{i}] {post['created_date']}")
        print(f"Title: {post['title']}")
        print(f"Sentiment: {post['sentiment']} (-1=neg, 0=neutral, 1=pos)")
        print(f"URL: {post['url']}")
