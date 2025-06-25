from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd

from nltk.tokenize import word_tokenize

from .constants import SENTIMENT_LABELS


def get_sentiment_frequency(df: pd.DataFrame, sentiment: str,
                            sentiment_labels: List[str] = SENTIMENT_LABELS,
                            return_as_tuple: bool = False):
    if sentiment.lower() not in sentiment_labels:
        raise ValueError(f"{sentiment} is not a valid sentiment label. Valid sentiments are {SENTIMENT_LABELS}")

    df_sentiment = df[df["sentiment"] == sentiment]

    if return_as_tuple:
        return df_sentiment, df_sentiment.shape[0]

    return df_sentiment.shape[0]


def get_sentiment_distribution(df: pd.DataFrame, sentiment_counts: Dict[str, int]):
    sentiment_labels = list(sentiment_counts.keys())
    sentiment_counts = list(sentiment_counts.values())

    total_posts = len(df)
    sentiment_percentages = [count * 100 / total_posts for count in sentiment_counts]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sentiment_labels, sentiment_counts, color=["red", "orange", "gray", "lightgreen", "green"])

    for bar, pct in zip(bars, sentiment_percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha='center', va='bottom')

    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Posts")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def normalize_text(text, lemmatizer, lemmatizer_func,
                   negation_words, punctuation_words, stop_words):
    tokens = word_tokenize(text)

    lemmatize = getattr(lemmatizer, lemmatizer_func)
    normalized = []

    prefix_NOT = False
    for token in tokens:
        if token in negation_words:
            prefix_NOT = True
            normalized.append(token)  # keep the negation word itself
            continue

        if token in punctuation_words:
            prefix_NOT = False
            normalized.append(token)
            continue

        if token in stop_words:
            continue

        token = lemmatize(token)
        if prefix_NOT:
            token = f"NOT_{token}"
        normalized.append(token)

    return normalized
