import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

from datasets import Dataset

# --- Load dataset ---
df = pd.read_csv("training.csv").dropna(subset=["text", "label"])

# Map labels -1,0,1 -> 0,1,2
label_map = {-1: 0, 0: 1, 1: 2}
df["labels"] = df["label"].map(label_map)

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
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

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

# --- Map back to original labels -1,0,1 ---
reverse_map = {0: -1, 1: 0, 2: 1}
final_outputs = [reverse_map[i] for i in predictions]

# --- Output distribution ---
import numpy as np
unique, counts = np.unique(final_outputs, return_counts=True)
output_summary = dict(zip(unique, counts))
print("Prediction distribution (0=neutral, 1=positive, -1=negative):")
print(output_summary)

