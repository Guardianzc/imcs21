import pandas as pd
import json
import os
from collections import defaultdict
import re


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


prefix = '../../../dataset'
# prefix = 'dataset'
split = defaultdict(list)
for _, row in pd.read_csv(os.path.join(prefix, 'split.csv'))[['example_id', 'split']].iterrows():
    split[row['split']].append(row['example_id'])

train = load_json(os.path.join(prefix, 'train.json'))
test = load_json(os.path.join(prefix, 'test.json'))
train_set = {pid: sample for pid, sample in train.items() if int(pid) in split['train']}
dev_set = {pid: sample for pid, sample in train.items() if int(pid) in split['dev']}
test_set = {pid: sample for pid, sample in test.items() if int(pid) in split['test']}


def make_data(samples, path, mode='train'):
    lines = ''
    with open(path, 'w', encoding='utf-8') as f:
        for pid, sample in samples.items():
            content = []
            for sent in sample['dialogue']:
                if sent['dialogue_act'] != 'Other':
                    content.append(sent['speaker'] + sent['sentence'])
            content = ''.join(content)
            title1, title2 = sample['report'][0], sample['report'][1]
            title1 = re.sub(r'[(]\d[)]', '', re.sub('\n', '', title1))
            title2 = re.sub(r'[(]\d[)]', '', re.sub('\n', '', title2))
            if mode == 'test':
                lines += content + '\n'
            else:
                lines += title1 + '\t' + content + '\n'
                lines += title2 + '\t' + content + '\n'
        f.write(lines)


make_data(train_set, 'data/train.tsv', mode='train')
make_data(dev_set, 'data/dev.tsv', mode='dev')
make_data(test_set, 'data/predict.tsv', mode='test')
