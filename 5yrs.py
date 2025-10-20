import requests
import datetime as dt
import time

def fetch_old_posts(company, subreddit, year=2020, max_posts=500):
    #Change the timestamps here
    after = int(dt.datetime(year, 1, 1).timestamp())
    before = int(dt.datetime(year + 1, 1, 1).timestamp())

    all_posts = []
    last_created_utc = before  # start from newest, move backward

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

            # --- Skip deleted/removed/empty posts ---
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

        # Update oldest timestamp to go further back
        last_created_utc = data[-1]["created_utc"]
        print(f"Collected {len(all_posts)} valid posts so far...")
        time.sleep(1)  # respect API rate limit

    if all_posts:
        oldest = min(all_posts, key=lambda x: x["created_utc"])
        newest = max(all_posts, key=lambda x: x["created_utc"])
        print(f"\nFetched {len(all_posts)} valid posts from r/{subreddit} in {year}")
        print(f"Oldest post: {dt.datetime.fromtimestamp(oldest['created_utc'])}")
        print(f"Newest post: {dt.datetime.fromtimestamp(newest['created_utc'])}")
    else:
        print("No valid posts found.")

    return all_posts


# ---- Example Run ----
if __name__ == "__main__":
    posts_2020 = fetch_old_posts("Amazon", "wallstreetbets", year=2020, max_posts=300)

    # Print first few posts with URLs
    for i, post in enumerate(posts_2020[:5], 1):
        created = dt.datetime.fromtimestamp(post["created_utc"])
        print(f"\n[{i}] {created}")
        print(f"Title: {post['title']}")
        print(f"Body: {post['body'][:200]}...")
        print(f"URL: {post['url']}")


