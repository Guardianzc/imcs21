import json
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True, type=str)
parser.add_argument("--gold", required=True, type=str)
parser.add_argument("--pred", required=True, type=str)
args = parser.parse_args()


# load normalized symptom
prefix = '../../dataset'
sym2id = {value: key for key, value in pd.read_csv(os.path.join(prefix, 'symptom_norm.csv'))['norm'].items()}
num_labels = len(sym2id)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def make_label(symptoms, target):
    if target == 'exp':
        label = [0] * num_labels
        for sx in symptoms:
            if sym2id.get(sx) is not None:
                label[sym2id.get(sx)] = 1
    else:
        label = [0] * (num_labels * 3)
        for sx_norm, sx_type in symptoms.items():
            if sym2id.get(sx_norm) is not None:
                label[sym2id.get(sx_norm) * 3 + int(sx_type)] = 1
    return label


# gold_data = load_json('dataset/test.json')
# pred_data = load_json('task/SRI/MTL-SRI/mtl_imp_pred.json')
gold_data = load_json(args.gold)
pred_data = load_json(args.pred)
golds, preds = [], []

for pid, sample in gold_data.items():
    gold = sample['implicit_info']['Symptom']
    pred = pred_data.get(pid)
    golds.append(make_label(gold, args.target))
    preds.append(make_label(pred, args.target))

golds, preds = np.array(golds), np.array(preds)

print('Exact Match Ratio: {}'.format(accuracy_score(golds, preds, normalize=True, sample_weight=None)))
print('Hamming loss: {}'.format(hamming_loss(golds, preds)))
print('Recall: {}'.format(precision_score(y_true=golds, y_pred=preds, average='samples', zero_division=0)))
print('Precision: {}'.format(recall_score(y_true=golds, y_pred=preds, average='samples', zero_division=0)))
print('F1 Measure samples: {}'.format(f1_score(y_true=golds, y_pred=preds, average='samples', zero_division=0)))
print('F1 Measure micro: {}'.format(f1_score(y_true=golds, y_pred=preds, average='micro', zero_division=0)))  # ours
print('F1 Measure macro: {}'.format(f1_score(y_true=golds, y_pred=preds, average='macro', zero_division=0)))
