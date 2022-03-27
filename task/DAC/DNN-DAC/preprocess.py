import pandas as pd
import json
import os
from collections import defaultdict


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


prefix = '../../../dataset'
# prefix = 'dataset'
saved_path = 'data'
os.makedirs(saved_path, exist_ok=True)

split = defaultdict(list)
for _, row in pd.read_csv(os.path.join(prefix, 'split.csv'))[['example_id', 'split']].iterrows():
    split[row['split']].append(row['example_id'])

train = load_json(os.path.join(prefix, 'train.json'))
test = load_json(os.path.join(prefix, 'test.json'))

train_set = {pid: sample for pid, sample in train.items() if int(pid) in split['train']}
dev_set = {pid: sample for pid, sample in train.items() if int(pid) in split['dev']}
test_set = {pid: sample for pid, sample in test.items() if int(pid) in split['test']}

tags = set()
for pid, sample in train_set.items():
    for _sent in sample['dialogue']:
        tags.add(_sent['dialogue_act'])
tags = list(tags)
tag2id = {tag: idx for idx, tag in enumerate(tags)}

with open(os.path.join(saved_path, 'class.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(tags))


def make_data(samples, save_path):
    out = ''
    for pid, sample in samples.items():
        for sent in sample['dialogue']:
            x = sent['speaker'] + '：' + sent['sentence']
            y = tag2id.get(sent['dialogue_act'])
            out += x + '\t' + str(y) + '\n'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out


make_data(train_set, os.path.join(saved_path, 'train.txt'))
make_data(dev_set, os.path.join(saved_path, 'dev.txt'))
make_data(test_set, os.path.join(saved_path, 'test.txt'))
