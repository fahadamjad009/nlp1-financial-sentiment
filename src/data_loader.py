"""
Data loader for FinancialPhraseBank sentiment dataset.
Uses parquet mirror compatible with datasets v5.
"""

from datasets import load_dataset
import pandas as pd
import os

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_financial_phrasebank():
    """Load FinancialPhraseBank from parquet-based HuggingFace repo."""
    print("Downloading FinancialPhraseBank dataset...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    return dataset


def explore_dataset(dataset):
    """Print basic stats about the dataset."""
    df = pd.DataFrame(dataset["train"])
    print("\n--- Dataset Overview ---")
    print("Columns:", df.columns.tolist())
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nSample rows:")
    print(df.head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    dataset = load_financial_phrasebank()
    df = explore_dataset(dataset)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/phrasebank_raw.csv", index=False)
    print("\nSaved to data/phrasebank_raw.csv")