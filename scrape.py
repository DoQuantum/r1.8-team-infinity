import praw
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

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
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    predicted_indices = torch.argmax(probabilities, dim=1)

    predictions = []
    for i, idx in enumerate(predicted_indices):
        label = sentiment_labels[idx]
        num_value = sentiment_map[label]
        predictions.append(num_value)
        print(f"Post {i+1}: {label} ({num_value}) -> {txt[i][:80]}...")

    # ---- Kaggle dataset evaluation with batching ----
    kaggle_df = pd.read_csv("kaggle_sentiment_data.csv")
    kaggle_df['numeric_sentiment'] = kaggle_df['analysis'].str.lower().map(sentiment_map)
    kaggle_texts = kaggle_df['body'].astype(str).tolist()
    ground_truth = kaggle_df['numeric_sentiment'].tolist()

    batch_size = 16
    kaggle_pred_numeric = []

    for i in range(0, len(kaggle_texts), batch_size):
        batch_texts = kaggle_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        batch_indices = torch.argmax(probs, dim=1)
        batch_preds = [sentiment_map[sentiment_labels[idx]] for idx in batch_indices]
        kaggle_pred_numeric.extend(batch_preds)

    acc = accuracy_score(ground_truth, kaggle_pred_numeric)
    prec = precision_score(ground_truth, kaggle_pred_numeric, average='macro', zero_division=0)

    print(f"\nKaggle Dataset Evaluation:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (macro): {prec:.3f}")

if __name__ == "__main__":
    main()
