"""
Fine-tune DistilBERT on financial sentiment classification.
Dataset: Twitter Financial News Sentiment (3-class)
Labels: 0=bearish, 1=neutral, 2=bullish
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report, confusion_matrix
import json

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/sentiment"
NUM_LABELS = 3
LABEL_MAP = {0: "bearish", 1: "neutral", 2: "bullish"}
ID2LABEL = {0: "bearish", 1: "neutral", 2: "bullish"}
LABEL2ID = {"bearish": 0, "neutral": 1, "bullish": 2}


def load_data():
    print("Loading dataset...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(
        labels, predictions,
        target_names=["bearish", "neutral", "bullish"],
        output_dict=True
    )
    return {
        "accuracy": report["accuracy"],
        "f1_bearish": report["bearish"]["f1-score"],
        "f1_neutral": report["neutral"]["f1-score"],
        "f1_bullish": report["bullish"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
    }


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    dataset = load_data()
    tokenized = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir="models/sentiment/logs",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\nSaving model and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nFinal evaluation:")
    results = trainer.evaluate()
    print(results)

    with open("models/sentiment/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Model saved to models/sentiment/")
    return results


if __name__ == "__main__":
    train()