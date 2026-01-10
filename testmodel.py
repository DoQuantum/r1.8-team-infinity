import requests
import datetime as dt
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import time, random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _session():
    s = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.6,
        status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    return s

def _month_ranges(year: int):
    cur = dt.datetime(year, 1, 1)
    while cur.year == year:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)  # next month
        yield int(cur.timestamp()), int(nxt.timestamp())
        cur = nxt

model_name_or_path = "./finbert_finetuned"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

reverse_map = {0: -1, 1: 0, 2: 1}

def fetch_old_posts(company, subreddit, year=2020, max_posts=500):
    company = company.upper()
    s = _session()
    headers = {"User-Agent": "Mozilla/5.0 (reddit-sentiment/1.0)"}

    all_posts = []
    print(f"Fetching posts from r/{subreddit} in {year} for '{company}'...")

    # Month-by-month reduces timeouts/520s a lot
    for after, before in _month_ranges(year):
        if len(all_posts) >= max_posts:
            break

        # Keep it simple for the backend: search title OR selftext
        # (PullPush supports title/selftext params) :contentReference[oaicite:2]{index=2}
        remaining = max_posts - len(all_posts)
        size = min(50, remaining)  # smaller than 100 = gentler

        url = (
            "https://api.pullpush.io/reddit/search/submission/"
            f"?subreddit={subreddit}"
            f"&after={after}&before={before}"
            f"&sort=desc&sort_type=created_utc&size={size}"
            f"&q={company}%20OR%20%24{company}"
        )

        try:
            resp = s.get(url, headers=headers, timeout=120)
        except requests.RequestException as e:
            print(f"Month window failed ({dt.datetime.fromtimestamp(after).date()}): {e}")
            continue

        if resp.status_code != 200:
            print(f"Month window failed ({dt.datetime.fromtimestamp(after).date()}): status {resp.status_code}")
            continue

        data = resp.json().get("data", [])
        if not data:
            continue

        for post in data:
            title = (post.get("title") or "").strip()
            body = (post.get("selftext") or "").strip()  # allow empty selftext
            if not title or title.lower() in ("[deleted]", "[removed]"):
                continue
            if body.lower() in ("[deleted]", "[removed]"):
                body = ""

            all_posts.append({
                "title": title,
                "body": body,
                "created_utc": int(post.get("created_utc") or 0),
                "url": f"https://www.reddit.com{post.get('permalink', '')}",
            })

        print(f"Collected {len(all_posts)} posts so far...")
        time.sleep(1.5 + random.uniform(0, 2.0))  # jitter helps

    return all_posts

def predict_sentiment(posts):
    for post in posts:
        text = (post.get("title", "") + " " + post.get("body", "")).strip()
        if not text:
            post["sentiment"] = 0
            continue

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=1).item()

        post["sentiment"] = reverse_map.get(pred_label, 0)

    return posts

if __name__ == "__main__":
    company_name = "GOOG"
    posts_2020 = fetch_old_posts(company_name, "wallstreetbets", year=2020, max_posts=20)
    scored_posts = predict_sentiment(posts_2020)

    if not scored_posts:
        print("No posts collected/scored. Nothing to save.")
        raise SystemExit(0)

    df = pd.DataFrame(scored_posts)

    # safer: create created_date from the column, not by mutating dicts
    df["created_date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce") \
                           .dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["created_date", "title", "body", "sentiment", "url"]
    df = df.reindex(columns=cols)

    output_path = f"{company_name}_sentiments.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved all {len(df)} posts with sentiment scores to {output_path}")

    # .gitignore update (same as your logic)
    gitignore_path = ".gitignore"
    try:
        with open(gitignore_path, "a+", encoding="utf-8") as f:
            f.seek(0)
            lines = f.read().splitlines()
            if output_path not in lines:
                f.write(f"\n{output_path}\n")
                print(f"Added '{output_path}' to .gitignore.")
            else:
                print(f"'{output_path}' is already in .gitignore.")
    except Exception as e:
        print(f"Warning: Could not update .gitignore — {e}")

    print(df.head())

    for i, row in df.head(5).iterrows():
        print(f"\n[{i+1}] {row['created_date']}")
        print(f"Title: {row['title']}")
        print(f"Sentiment: {row['sentiment']} (0=neutral, 1=positive, -1=negative)")
        print(f"URL: {row['url']}")
