import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import time
import random
import datetime as dt
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

SPREADSHEET_PATH = "./Portfolio Analysis and Optimization Suggestions.xlsx"
SUBREDDIT = "wallstreetbets"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
MAX_POSTS_PER_TICKER = 800

# NEW constraints
MIN_TICKER_POSTS = 20
TICKER_QUERY_TIMEOUT_SEC = 120  # 2 minutes

OUT_DIR = Path("outputs_pullpush_finbert")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "./finbert_finetuned"

def _session():
    s = requests.Session()
    retry = Retry(
        total=10,
        backoff_factor=1.6,
        status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    return s

def _to_ts(date_str: str) -> int:
    return int(dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.timezone.utc).timestamp())

START_TS = _to_ts(START_DATE)
END_TS = _to_ts(END_DATE) + 1

def load_ticker_company_pairs(path: str):
    df = pd.read_excel(path)

    ticker_col = None
    name_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if ticker_col is None and ("ticker" in col_lower or "symbol" in col_lower):
            ticker_col = col
        if name_col is None and ("company" in col_lower or "name" in col_lower):
            name_col = col

    if ticker_col is None:
        raise ValueError("No ticker column found.")
    if name_col is None:
        raise ValueError("No company name column found.")

    df = df[[ticker_col, name_col]].dropna()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
    df[name_col] = df[name_col].astype(str).str.strip()

    pairs = []
    seen = set()
    for t, n in zip(df[ticker_col], df[name_col]):
        if t in {"N/A", "NA", "NONE", "NULL", ""}:
            continue
        key = (t, n)
        if key not in seen:
            seen.add(key)
            pairs.append(key)

    print("Loaded pairs:", len(pairs))
    print("First 10:", pairs[:10])
    return pairs

def fetch_posts_pullpush(
    query: str,
    ticker_label: str,
    subreddit: str,
    after_ts: int,
    before_ts: int,
    max_posts=None,
    max_seconds=None,  # NEW
):
    s = _session()
    headers = {"User-Agent": "reddit-sentiment/1.0 (pullpush)"}

    t0 = time.monotonic()
    all_posts = []
    cur_before = before_ts

    while True:
        if max_seconds is not None and (time.monotonic() - t0) > max_seconds:
            break

        if max_posts is not None and len(all_posts) >= max_posts:
            break

        remaining = (max_posts - len(all_posts)) if max_posts is not None else 100
        size = min(100, remaining)

        q = requests.utils.quote(query)
        url = (
            "https://api.pullpush.io/reddit/search/submission/"
            f"?subreddit={subreddit}"
            f"&after={after_ts}&before={cur_before}"
            f"&sort=desc&sort_type=created_utc&size={size}"
            f"&q={q}"
        )

        try:
            resp = s.get(url, headers=headers, timeout=90)
        except requests.RequestException:
            time.sleep(2.0 + random.uniform(0, 2.0))
            continue

        if resp.status_code == 429:
            time.sleep(3.0 + random.uniform(1.0, 3.0))
            continue

        if resp.status_code != 200:
            time.sleep(2.0 + random.uniform(0, 2.0))
            continue

        data = resp.json().get("data", [])
        if not data:
            break

        oldest = None
        for post in data:
            title = (post.get("title") or "").strip()
            body = (post.get("selftext") or "").strip()

            if not title or title.lower() in ("[deleted]", "[removed]"):
                continue
            if body.lower() in ("[deleted]", "[removed]"):
                body = ""

            created = int(post.get("created_utc") or 0)
            if created <= 0:
                continue

            permalink = post.get("permalink", "") or ""
            all_posts.append({
                "ticker": ticker_label,
                "title": title,
                "body": body,
                "created_utc": created,
                "url": f"https://www.reddit.com{permalink}",
            })

            if oldest is None or created < oldest:
                oldest = created

        if oldest is None:
            break

        cur_before = oldest - 1
        if cur_before <= after_ts:
            break

        time.sleep(0.4 + random.uniform(0, 0.6))

    return all_posts

def _dedupe_posts(posts):
    seen = set()
    out = []
    for p in posts:
        key = (p.get("created_utc"), p.get("url"))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def fetch_with_constraints(ticker: str, company_name: str):
    ticker = str(ticker).strip().upper()
    company_name = str(company_name).strip()

    # 2) If ticker queries take > 2 minutes, switch to full name
    t0 = time.monotonic()
    posts_t = fetch_posts_pullpush(
        ticker, ticker, SUBREDDIT, START_TS, END_TS, MAX_POSTS_PER_TICKER,
        max_seconds=TICKER_QUERY_TIMEOUT_SEC
    )
    elapsed = time.monotonic() - t0

    if elapsed > TICKER_QUERY_TIMEOUT_SEC:
        posts_name = fetch_posts_pullpush(company_name, ticker, SUBREDDIT, START_TS, END_TS, MAX_POSTS_PER_TICKER)
        return posts_name, "name_timeout"

    remaining_time = max(1.0, TICKER_QUERY_TIMEOUT_SEC - elapsed)

    t1 = time.monotonic()
    posts_dollar = fetch_posts_pullpush(
        f"${ticker}", ticker, SUBREDDIT, START_TS, END_TS, MAX_POSTS_PER_TICKER,
        max_seconds=remaining_time
    )
    elapsed2 = time.monotonic() - t1

    if (elapsed + elapsed2) > TICKER_QUERY_TIMEOUT_SEC:
        posts_name = fetch_posts_pullpush(company_name, ticker, SUBREDDIT, START_TS, END_TS, MAX_POSTS_PER_TICKER)
        return posts_name, "name_timeout"

    combined = _dedupe_posts(posts_t + posts_dollar)

    # 1) If < 20 posts from ticker queries, use full name
    if len(combined) < MIN_TICKER_POSTS:
        posts_name = fetch_posts_pullpush(company_name, ticker, SUBREDDIT, START_TS, END_TS, MAX_POSTS_PER_TICKER)
        posts_name = _dedupe_posts(posts_name)
        return posts_name, "name_lowcount"

    return combined, "ticker_ok"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

reverse_map = {0: -1, 1: 0, 2: 1}

def predict_sentiment(posts, batch_size=16, max_length=192):
    texts = []
    idxs = []
    for i, p in enumerate(posts):
        text = (p.get("title", "") + " " + p.get("body", "")).strip()
        if not text:
            p["sentiment"] = 0
        else:
            texts.append(text)
            idxs.append(i)

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1).tolist()

        for j, lab in enumerate(preds):
            post_i = idxs[start + j]
            posts[post_i]["sentiment"] = reverse_map.get(lab, 0)

    return posts

if __name__ == "__main__":
    ticker_company_pairs = load_ticker_company_pairs(SPREADSHEET_PATH)
    # IMPORTANT: do NOT slice to [:1] unless testing
    # ticker_company_pairs = ticker_company_pairs[:1]

    all_rows = []

    for tkr, company_name in ticker_company_pairs:
        try:
            print("\nProcessing:", tkr, "|", company_name)

            posts, mode = fetch_with_constraints(tkr, company_name)
            posts = _dedupe_posts(posts)

            print(tkr, "total fetched:", len(posts), "| mode:", mode)

            if not posts:
                continue

            posts = predict_sentiment(posts, batch_size=16)

            df = pd.DataFrame(posts)
            df["created_date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df.reindex(columns=["ticker", "created_date", "title", "body", "sentiment", "url"])

            out_path = OUT_DIR / f"{tkr}_sentiments.csv"
            df.to_csv(out_path, index=False, encoding="utf-8")
            print("Saved:", out_path)

            all_rows.append(df)

        except Exception as e:
            print("FAILED:", tkr, "|", company_name, "|", repr(e))
            continue

    if all_rows:
        big = pd.concat(all_rows, ignore_index=True)
        combined_path = OUT_DIR / "ALL_TICKERS_sentiments.csv"
        big.to_csv(combined_path, index=False, encoding="utf-8")
        print("Saved combined:", combined_path)

    try:
        gi = Path(".gitignore")
        gi.touch(exist_ok=True)
        lines = gi.read_text(encoding="utf-8").splitlines()
        entry = str(OUT_DIR) + "/"
        if entry not in lines:
            with gi.open("a", encoding="utf-8") as f:
                f.write(f"\n{entry}\n")
    except Exception:
        pass