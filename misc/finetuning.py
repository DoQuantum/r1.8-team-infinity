import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, create_optimizer

# --- Load dataset ---
df = pd.read_csv("training.csv").dropna(subset=["text", "label"])

# --- Map labels -1,0,1 -> 0,1,2 for training ---
label_map = {-1: 0, 0: 1, 1: 2}
df["labels"] = df["label"].map(label_map)

# --- Load tokenizer and model ---
model_name = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# --- Tokenize and pad the entire dataset ---
tokenized = tokenizer(
    list(df["text"]),
    truncation=True,
    padding=True,
    return_tensors="np"
)
input_ids = tokenized["input_ids"]
attention_mask = tokenized["attention_mask"]
labels = df["labels"].to_numpy()

# --- Train/test split (manual without sklearn) ---
num_samples = len(labels)
split_idx = int(num_samples * 0.9)  # 90% train, 10% test

train_features = {
    "input_ids": input_ids[:split_idx],
    "attention_mask": attention_mask[:split_idx]
}
train_labels = labels[:split_idx]

test_features = {
    "input_ids": input_ids[split_idx:],
    "attention_mask": attention_mask[split_idx:]
}
test_labels = labels[split_idx:]

# --- Convert to tf.data.Dataset ---
batch_size = 8
train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)) \
    .shuffle(len(train_labels)).batch(batch_size)
test_tf_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)) \
    .batch(batch_size)

# --- Optimizer ---
num_train_steps = len(train_tf_dataset) * 3  # 3 epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps
)

# --- Compile model ---
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])

# --- Train ---
model.fit(train_tf_dataset, validation_data=test_tf_dataset, epochs=3)

# --- Save fine-tuned model ---
model.save_pretrained("./finbert_reddit_finetuned")
tokenizer.save_pretrained("./finbert_reddit_finetuned")

# --- Predict and map back to -1,0,1 ---
preds = model.predict(test_tf_dataset).logits
predicted_labels = np.argmax(preds, axis=1)
reverse_map = {0: -1, 1: 0, 2: 1}
final_outputs = [reverse_map[i] for i in predicted_labels]

# --- Output prediction distribution ---
unique, counts = np.unique(final_outputs, return_counts=True)
output_summary = dict(zip(unique, counts))
print("Prediction distribution (0=neutral, 1=positive, -1=negative):")
print(output_summary)

