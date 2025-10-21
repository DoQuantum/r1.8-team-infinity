import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from datasets import Dataset
import numpy as np

# --- Load dataset ---
df = pd.read_csv("training.csv").dropna(subset=["text", "label"])

# Use labels as-is (0=negative, 1=neutral, 2=positive or whatever your convention is)
df["labels"] = df["label"].astype(int)

# --- Create Hugging Face Dataset ---
dataset = Dataset.from_pandas(df[["text", "labels"]])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# --- Tokenizer ---
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

train_dataset = dataset["train"].map(tokenize, batched=True)
test_dataset = dataset["test"].map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# --- DataLoader ---
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# --- Load model ---
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# --- Optimizer ---
optimizer = AdamW(model.parameters(), lr=2e-5)

# --- Training loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} completed — Avg Loss: {total_loss / len(train_loader):.4f}")

# --- Evaluation ---
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# --- Output distribution ---
unique, counts = np.unique(predictions, return_counts=True)
output_summary = dict(zip(unique, counts))
print("\nPrediction distribution (0, 1, 2):")
print(output_summary)

