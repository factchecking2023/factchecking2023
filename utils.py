# -*- coding: utf-8 -*-
from datetime import datetime

import torch
import pandas as pd
import re
import math
import json
from sklearn.metrics import classification_report

def cred(days, rid, pr):
    w1 = 0.4  # PageRank
    w2 = 0.4  # RefId
    w3 = 0.2  # Days
    if days <= 0:
        days = 3650
    cred_score = w1 * pr / 10.0 + w2 * math.exp(- (rid - 1)) + w3 * math.exp(- days / 30.0)
    return cred_score

def remove_dup_space(text):
    clean_text = re.sub(r'(\s)\s+', r'\1', text)
    return clean_text

def get_limit_text(text, limit = 512):
    cnt = 0
    for i in range(len(text)):
        if text[i] in ['\t', '\n', ' ']:
            cnt += 1
        if cnt >= limit:
            return text[:i]
    return text

def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)

def my_read_csv(path, sep=","):
    df = pd.read_csv(path, sep=sep)
    df = df.dropna(subset=['label', 'claim', 'main_text'])
    df = df.fillna('')
    df = df[ df['label'].isin(['unproven', 'true', 'false', 'mixture']) ]

    df["claim"] = df["claim"].astype(str)
    df["main_text"] = df["main_text"].astype(str)
    df["claim_entities_desc"] = df["claim_entities_desc"].astype(str)
    df["text_entities_desc"] = df["text_entities_desc"].astype(str)
    df["label"] = df["label"].astype(str)
    
    df = df[["claim_id", "claim", "main_text", "claim_entities_desc", "text_entities_desc", "label"]]
    return df

def evaluate(hparams, pred_path):
    with open(pred_path) as f:
        predictions = json.loads(f.read())
    y_pred = [o["predicted_label"] for o in predictions]
    y_true = [s["label"] for s in predictions]
    print(classification_report(y_true, y_pred, digits=4))