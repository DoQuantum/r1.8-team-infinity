import praw
import os
from dotenv import load_dotenv

load_dotenv()

Client_ID = os.getenv('CLIENT_ID')
Client_Secret = os.getenv('CLIENT_SECRET')
User_Agent = os.getenv('USER_AGENT')
reddit = praw.Reddit(
    client_id=Client_ID,
    client_secret=Client_Secret,
    user_agent=User_Agent,
)

for submission in reddit.subreddit("wallstreetbets").hot(limit=50):
    title = print(submission.title)
    id = print(submission.id)
    body = print(submission.selftext)


