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


def make_data(samples, prefix, mode='train'):
    src, tgt = '', ''
    for pid, sample in samples.items():
        content = []
        for sent in sample['dialogue']:
            if sent['dialogue_act'] != 'Other':
                content.append(sent['speaker'] + sent['sentence'])
        content = ''.join(content)
        title1, title2 = sample['report'][0], sample['report'][1]
        title1 = re.sub(r'[(]\d[)]', '', re.sub('\n', '', title1))
        title2 = re.sub(r'[(]\d[)]', '', re.sub('\n', '', title2))
        content = ' '.join(content)
        title1 = ' '.join(title1)
        title2 = ' '.join(title2)
        if mode == 'test':
            src += content + '\n'
            tgt += title1 + '\t' + title2 + '\n'
        else:
            src += content + '\n' + content + '\n'
            tgt += title1 + '\n' + title2 + '\n'
    with open(os.path.join(prefix, 'src-{}.txt'.format(mode)), 'w', encoding='utf-8') as f:
        f.write(src)
    with open(os.path.join(prefix, 'tgt-{}.txt'.format(mode)), 'w', encoding='utf-8') as f:
        f.write(tgt)


make_data(train_set, 'data', mode='train')
make_data(dev_set, 'data', mode='dev')
make_data(test_set, 'data', mode='test')
