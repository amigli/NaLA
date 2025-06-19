import math
from collections import defaultdict
from typing import List

import pandas as pd
from tqdm import tqdm


def train_naive_bayes(D: pd.DataFrame, C: List[str]):
    n_doc = D.shape[0]  # total number of documents
    log_prior = {}  # prior probabilities (in log scale)
    log_likelihood = {}  # likelihood of each word given a class (in log scale)
    V = set()  # vocabulary

    # Count word frequencies per class
    word_counts = {c: defaultdict(int) for c in C}
    class_word_totals = {c: 0 for c in C}

    for c in C:
        # Subset of documents for class c
        class_docs = D[D["sentiment"] == c].drop(columns="sentiment")

        # Compute prior
        n_class = class_docs.shape[0]
        log_prior[c] = math.log(n_class / n_doc)

        # Count words in all docs of class c
        for _, row in tqdm(class_docs.iterrows(), desc=f"Counting words for class {c}", total=n_class):
            for word in row.text:
                # apply binary count: count the word one-time per document
                word_counts[c][word] += 1
                class_word_totals[c] += 1
                V.add(word)

    V = list(V)
    vocab_size = len(V)

    # Compute log likelihood with Laplace smoothing
    for c in tqdm(C, desc=f"Computing log-likelihood"):
        log_likelihood[c] = {}
        for word in V:
            count_w_c = word_counts[c].get(word, 0)
            log_likelihood[c][word] = math.log((count_w_c + 1) / (class_word_totals[c] + vocab_size))

    return log_prior, log_likelihood, V


def test_naive_bayes(test_D: pd.DataFrame, log_prior, log_likelihood, V, C: List[str]):
    predictions = []
    V = set(V)  # faster lookup

    for _, row in tqdm(test_D.iterrows(), total=len(test_D), desc="Testing Naive Bayes"):
        doc_log_prob = {}

        for c in C:
            log_prob = log_prior[c]
            for word in row.text:
                if word in V:
                    log_prob += log_likelihood[c].get(word, 0)  # word may be missing from this class
            doc_log_prob[c] = log_prob

        # Choose the class with the highest log probability
        predicted_class = max(doc_log_prob, key=doc_log_prob.get)
        predictions.append(predicted_class)

    return predictions
