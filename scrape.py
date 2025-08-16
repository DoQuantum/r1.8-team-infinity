import praw
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def main():
    load_dotenv()
    Client_ID = os.getenv('CLIENT_ID')
    Client_Secret = os.getenv('CLIENT_SECRET')
    User_Agent = os.getenv('USER_AGENT')

    reddit = praw.Reddit(
        client_id=Client_ID,
        client_secret=Client_Secret,
        user_agent=User_Agent,
    )

    results = []
    for submission in reddit.subreddit("wallstreetbets").hot(limit=10):
        results.append({"title": submission.title, "body": submission.selftext})

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    txt = [item["title"] + " " + item["body"] for item in results]
    inputs = tokenizer(txt, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)

    sentiment_labels = ["negative", "neutral", "positive"]
    predicted_indices = torch.argmax(probabilities, dim=1)

    for i, idx in enumerate(predicted_indices):
        print(f"Post {i+1}: {sentiment_labels[idx]}  ->  {txt[i][:80]}...")

if __name__ == "__main__":
    main()
