import os
import re
import time
import random
import datetime as dt

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------
# HTTP session w/ retries
# -----------------------
def _session():
    s = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.6,
        status_forcelist=[429, 500, 502, 503, 504, 520, 522, 524],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
    return s


# -----------------------
# Time helpers
# -----------------------
UTC = dt.timezone.utc

def quarter_range(time_period: str) -> tuple[int, int]:
    """
    '2020Q1' -> (after_ts, before_ts) where before is the first second of next quarter.
    """
    tp = str(time_period).strip().upper()
    year = int(tp[:4])
    q = int(tp[-1])

    start_month = 1 + 3 * (q - 1)
    start = dt.datetime(year, start_month, 1, tzinfo=UTC)

    if q == 4:
        end = dt.datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        end = dt.datetime(year, start_month + 3, 1, tzinfo=UTC)

    return int(start.timestamp()), int(end.timestamp())


# -----------------------
# Mention matching helper (for filtering false positives)
# -----------------------
def _mention_regex(ticker: str) -> re.Pattern:
    t = ticker.upper().strip()
    esc = re.escape(t)
    if re.fullmatch(r"[A-Z0-9]+", t):
        # normal tickers: GOOG, AAPL
        pat = rf"(?i)(?:\b{esc}\b|\${esc}\b)"
    else:
        # tickers with dots etc: BRK.B, BF.B
        pat = rf"(?i)(?:\${esc}\b|(?<!\w){esc}(?!\w))"
    return re.compile(pat)


# -----------------------
# PullPush fetch (single time window, paginated)
# -----------------------
def fetch_posts_in_range(company: str, subreddit: str, after_ts: int, before_ts: int, max_posts: int = 500, page_size: int = 50):
    company = company.upper().strip()
    s = _session()
    headers = {"User-Agent": "Mozilla/5.0 (reddit-sentiment/1.0)"}

    seen = set()
    posts = []
    token_pat = _mention_regex(company)

    cursor_before = before_ts

    while len(posts) < max_posts:
        remaining = max_posts - len(posts)
        size = min(page_size, remaining)

        params = {
            "subreddit": subreddit,
            "after": after_ts,
            "before": cursor_before,
            "sort": "desc",
            "sort_type": "created_utc",
            "size": size,
            "q": f"{company} OR ${company}",
        }

        try:
            resp = s.get("https://api.pullpush.io/reddit/search/submission/", headers=headers, params=params, timeout=(10, 90))
        except requests.RequestException:
            break

        if resp.status_code != 200:
            break

        data = resp.json().get("data", [])
        if not data:
            break

        added = 0
        oldest_ts = None

        for p in data:
            title = (p.get("title") or "").strip()
            body = (p.get("selftext") or "").strip()
            created_utc = int(p.get("created_utc") or 0)
            permalink = p.get("permalink", "") or ""
            url = f"https://www.reddit.com{permalink}"

            if not title or title.lower() in ("[deleted]", "[removed]"):
                continue
            if body.lower() in ("[deleted]", "[removed]"):
                body = ""

            if url in seen:
                continue

            text = f"{title} {body}".strip()
            if not token_pat.search(text):
                continue

            seen.add(url)
            posts.append({"title": title, "body": body, "created_utc": created_utc, "url": url})
            added += 1
            oldest_ts = created_utc if (oldest_ts is None or created_utc < oldest_ts) else oldest_ts

            if len(posts) >= max_posts:
                break

        if oldest_ts is None or added == 0:
            break

        cursor_before = oldest_ts - 1
        time.sleep(1.0 + random.uniform(0, 1.5))

    return posts


# -----------------------
# Model load + batched sentiment
# -----------------------
model_name_or_path = "./finbert_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

reverse_map = {0: -1, 1: 0, 2: 1}

def predict_sentiment(posts, batch_size: int = 16, max_length: int = 256):
    texts = [(p.get("title", "") + " " + p.get("body", "")).strip() for p in posts]

    with torch.inference_mode():
        for i in range(0, len(posts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=1).tolist()
            for j, pred_label in enumerate(preds):
                posts[i + j]["sentiment"] = reverse_map.get(pred_label, 0)

    return posts


# -----------------------
# Driver: read your CSV + write one output CSV per company with time_period column
# -----------------------
def run_from_company_quarters_csv(
    input_csv_path: str,
    subreddit: str = "wallstreetbets",
    max_posts_per_quarter: int = 200,
    output_dir: str = "sentiments_by_company",
    overwrite: bool = False,
):
    df_in = pd.read_csv(input_csv_path)

    # expected columns: time_period,date,company
    df_in["time_period"] = df_in["time_period"].astype(str).str.strip().str.upper()
    df_in["company"] = df_in["company"].astype(str).str.strip().str.upper()
    df_in = df_in.dropna(subset=["time_period", "company"])

    os.makedirs(output_dir, exist_ok=True)

    companies = sorted(df_in["company"].unique().tolist())

    # ignore folder once (instead of listing thousands of files)
    gitignore_path = ".gitignore"
    try:
        with open(gitignore_path, "a+", encoding="utf-8") as f:
            f.seek(0)
            lines = f.read().splitlines()
            if output_dir not in lines:
                f.write(f"\n{output_dir}\n")
    except Exception:
        pass

    for idx, company in enumerate(companies, start=1):
        out_path = os.path.join(output_dir, f"{company}_sentiments.csv")
        if os.path.exists(out_path) and not overwrite:
            print(f"[{idx}/{len(companies)}] {company}: exists, skipping ({out_path})")
            continue

        tps = sorted(df_in.loc[df_in["company"] == company, "time_period"].unique().tolist())

        all_posts = []
        print(f"\n[{idx}/{len(companies)}] {company}: quarters={len(tps)}")

        for tp in tps:
            after_ts, before_ts = quarter_range(tp)
            posts = fetch_posts_in_range(
                company=company,
                subreddit=subreddit,
                after_ts=after_ts,
                before_ts=before_ts,
                max_posts=max_posts_per_quarter,
                page_size=50,
            )
            for p in posts:
                p["time_period"] = tp
                p["company"] = company
            all_posts.extend(posts)

            print(f"  {tp}: +{len(posts)} (total {len(all_posts)})")
            time.sleep(0.8 + random.uniform(0, 1.2))

        if not all_posts:
            print(f"  No posts found for {company}. Writing empty CSV.")
            out_df = pd.DataFrame(columns=["time_period", "company", "created_date", "title", "body", "sentiment", "url"])
            out_df.to_csv(out_path, index=False, encoding="utf-8")
            continue

        scored = predict_sentiment(all_posts)

        out_df = pd.DataFrame(scored)
        out_df["created_date"] = pd.to_datetime(out_df["created_utc"], unit="s", errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

        cols = ["time_period", "company", "created_date", "title", "body", "sentiment", "url"]
        out_df = out_df.reindex(columns=cols)

        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  Saved {len(out_df)} rows -> {out_path}")

        # small pause between tickers
        time.sleep(1.5 + random.uniform(0, 2.0))


if __name__ == "__main__":
    INPUT = "tickers_unique.csv" 
    run_from_company_quarters_csv(
        input_csv_path=INPUT,
        subreddit="wallstreetbets",
        max_posts_per_quarter=200,     # adjust
        output_dir="sentiments_by_company",
        overwrite=False,
    )
