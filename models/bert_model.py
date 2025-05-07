import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Create required directories if they don't exist
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Configuration
SAMPLE_LIMIT = 5000
dataset_choice = os.getenv("DATASET_CHOICE", "1")

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset selection
if dataset_choice == "1":
    dataset_path = "data/imdb_IMDB Dataset.csv"
    text_column = "review"
    label_column = "sentiment"
elif dataset_choice == "2":
    dataset_path = "data/yelp_review.csv"
    text_column = "text"
    label_column = "sentiment"
elif dataset_choice == "3":
    dataset_path = "data/twitter_training.1600000.processed.noemoticon.csv"
    text_column = "text"
    label_column = "target"
else:
    dataset_path = "data/imdb_IMDB Dataset.csv"
    text_column = "review"
    label_column = "sentiment"

# Load dataset
if dataset_choice == "3":
    column_names = ["target", "id", "date", "flag", "user", "text"]
    df = pd.read_csv(dataset_path, encoding="latin1", names=column_names, header=None)
else:
    df = pd.read_csv(dataset_path)

# Drop missing values
df.dropna(inplace=True)

# Preview the data
print("\nPreview of the dataset (first 5 rows):")
print(df.head())

# Filter neutral labels (only for Yelp)
if dataset_choice == "2":
    df = df[df["sentiment"].str.lower().isin(["positive", "negative"])]

# Label mapping
df[text_column] = df[text_column].astype(str)
if dataset_choice == "3":
    df[label_column] = df[label_column].map({0: 0, 4: 1})
else:
    df[label_column] = df[label_column].str.lower().map({"positive": 1, "negative": 0})

df.dropna(subset=[label_column], inplace=True)

# Apply sample limitation after cleaning
if SAMPLE_LIMIT > 0 and SAMPLE_LIMIT < len(df):
    df = df.sample(SAMPLE_LIMIT, random_state=42)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[text_column].tolist(), df[label_column].tolist(), test_size=0.2, random_state=42
)

# Tokenize the input using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Prepare datasets for HuggingFace Trainer
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "label": train_labels
})
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "label": test_labels
})

# Load pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments for HuggingFace Trainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    save_strategy="no",
    disable_tqdm=False
)

# Initialize HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train and evaluate the model
trainer.train()
eval_result = trainer.evaluate()

# Print evaluation results
print("\nEvaluation Results:")
print(eval_result)

# Save accuracy and F1-score to a file for later plotting
with open("plots/bert_eval_results.txt", "w") as f:
    for k, v in eval_result.items():
        f.write(f"{k}: {v:.4f}\n")