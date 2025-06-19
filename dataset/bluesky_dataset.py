import string
from typing import Union, Optional

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from torch.utils.data import Dataset

from utils import normalize_text
from utils.constants import SENTIMENT_LABELS

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

STOPWORDS_EN = set(stopwords.words('english'))
NEGATION_WORDS = {"n't", "not", "no", "never"}
PUNCTUATION = set(string.punctuation)


class BlueSkyDataset(Dataset):
    def __init__(self, posts: Union[str, pd.DataFrame], vocab: Optional[dict] = None,
                 max_len: Optional[int] = None,
                 lemmatizer=None,
                 lemmatizer_func=None,
                 tokenizer=None,
                 pre_process_text: bool = True):
        super(BlueSkyDataset, self).__init__()

        if isinstance(posts, pd.DataFrame):
            self.data = posts
        else:
            self.data = pd.read_csv(posts, lineterminator='\n', encoding='utf-8')

        self.vocab = vocab
        self.max_len = max_len

        if self.vocab:
            self.pad_idx = vocab.get("<PAD>", 0)
            self.unk_idx = vocab.get("<UNK>", 1)

        self.lemmatizer = lemmatizer
        self.lemmatizer_func = lemmatizer_func
        self.tokenizer = tokenizer

        self.pre_process_text = pre_process_text

        if self.lemmatizer is None and self.tokenizer is None and self.pre_process_text:
            raise ValueError("Either lemmatizer or tokenizer must be provided")

        if self.lemmatizer and self.lemmatizer_func is None and self.pre_process_text:
            raise ValueError("Lemmatizer was provided but no lemmatizer_func is available. Specify one")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentiment_label = SENTIMENT_LABELS.index(row.sentiment.lower())

        if self.tokenizer is None and self.pre_process_text:
            tokens = normalize_text(row.text, self.lemmatizer, self.lemmatizer_func,
                                    NEGATION_WORDS, PUNCTUATION, STOPWORDS_EN)
            token_ids = [self.vocab.get(token, self.unk_idx) for token in tokens]

            # Pad or truncate
            if len(token_ids) < self.max_len:
                token_ids += [self.pad_idx] * (self.max_len - len(token_ids))
            else:
                token_ids = token_ids[:self.max_len]

            token_ids = torch.tensor(token_ids, dtype=torch.long)
        elif self.pre_process_text:
            token_ids = self.tokenizer(row.text, return_tensors='pt')
        else:
            token_ids = row.text

        return token_ids, torch.tensor(sentiment_label, dtype=torch.long)
