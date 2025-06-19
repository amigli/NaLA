from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm

from .constants import SEED
from .pre_processing import normalize_text


def get_device(device_preference: str = "auto") -> torch.device:
    if device_preference == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        try:
            return torch.device(device_preference)
        except Exception as e:
            raise ValueError(f"Invalid device specified: '{device_preference}'") from e


def stratified_downsample(df, target_col, frac=0.5, random_state=SEED):
    groups = df.groupby(target_col)
    sampled = [group.sample(frac=frac, random_state=random_state) for _, group in groups]
    return pd.concat(sampled).reset_index(drop=True)


def build_vocab(texts, lemmatizer, lemmatizer_func, negation_words, punctuation, stop_words, k=10_000,
                specials=["<PAD>", "<UNK>"]):
    counter = Counter()

    print("🔍 Building vocabulary...")
    for text in tqdm(texts, desc="Tokenizing texts"):
        tokens = normalize_text(text, lemmatizer, lemmatizer_func, negation_words, punctuation, stop_words)
        counter.update(tokens)

    # Most common K words
    most_common = counter.most_common(k)

    vocab = {token: idx for idx, token in enumerate(specials)}
    for token, _ in most_common:
        if token not in vocab:  # avoid duplicates with specials
            vocab[token] = len(vocab)

    print(f"✅ Vocab built with {len(vocab)} tokens (including specials)")
    return vocab
