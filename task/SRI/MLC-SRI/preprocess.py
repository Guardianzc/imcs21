import os
import json
import pandas as pd
from collections import defaultdict


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        if isinstance(data, list):
            print('writing {} records to {}'.format(len(data), path))


prefix = '../../../dataset'
# prefix = 'dataset/'

# load train/test.json
train = load_json(os.path.join(prefix, 'train.json'))
test = load_json(os.path.join(prefix, 'test_input.json'))

# load split.csv
split = defaultdict(list)
for _, row in pd.read_csv(os.path.join(prefix, 'split.csv'))[['example_id', 'split']].iterrows():
    split[row['split']].append(row['example_id'])

# make dataset for train/dev/test set, note that in test set, the label is empty
train_set = {pid: sample for pid, sample in train.items() if int(pid) in split['train']}
dev_set = {pid: sample for pid, sample in train.items() if int(pid) in split['dev']}
test_set = {pid: sample for pid, sample in test.items() if int(pid) in split['test']}

# load normalized symptom
sym2id = {value: key for key, value in pd.read_csv(os.path.join(prefix, 'symptom_norm.csv'))['norm'].items()}
num_labels = len(sym2id)


# make label
def make_label_exp(symptoms):
    label = [0] * num_labels
    for symptom in symptoms:
        if sym2id.get(symptom) is not None:
            label[sym2id.get(symptom)] = 1
    return label


def make_label_imp(symptom_norm, symptom_type):
    assert len(symptom_norm) == len(symptom_type)
    label = [0] * (num_labels * 3)
    for sx_norm, sx_type in zip(symptom_norm, symptom_type):
        if sym2id.get(sx_norm) is not None:
            label[sym2id.get(sx_norm) * 3 + int(sx_type)] = 1
    return label


# make train/dev/test set, extract input & output
# note: one can use all the information in the train set to build more complicated models
def make_dataset(samples, add: bool = True):
    exp_samples, imp_samples = [], []
    for pid, sample in samples.items():
        # explicit symptoms
        if 'diagnosis' in sample:
            y = make_label_exp(sample['explicit_info']['Symptom'])
        else:
            y = []
        exp_samples.append((sample['self_report'], y, pid, 10))
        # implicit symptoms
        for sent in sample['dialogue']:
            x = sent['speaker'] + ':' + sent['sentence']
            if 'symptom_norm' in sent and 'symptom_type' in sent:
                # train/dev set
                y = make_label_imp(sent['symptom_norm'], sent['symptom_type'])
                # let the sample have a greater probability to be sampled
                weight = 20 if sent['symptom_norm'] and sent['symptom_type'] else 1
                if add:
                    exp_samples.append((sent['sentence'], make_label_exp(sent['symptom_norm']), pid, 1))
            else:
                # test set
                y = []
                weight = 1
            imp_samples.append((x, y, pid, weight))
    return exp_samples, imp_samples


train_set_exp, train_set_imp = make_dataset(train_set)
dev_set_exp, dev_set_imp = make_dataset(dev_set, add=False)
test_set_exp, test_set_imp = make_dataset(test_set, add=False)


saved_path = 'sri_data'
os.makedirs(saved_path, exist_ok=True)

write_json(train_set_exp, os.path.join(saved_path, 'train_set_exp.json'))
write_json(dev_set_exp, os.path.join(saved_path, 'dev_set_exp.json'))
write_json(test_set_exp, os.path.join(saved_path, 'test_set_exp.json'))

write_json(train_set_imp, os.path.join(saved_path, 'train_set_imp.json'))
write_json(dev_set_imp, os.path.join(saved_path, 'dev_set_imp.json'))
write_json(test_set_imp, os.path.join(saved_path, 'test_set_imp.json'))

print('exp: train/dev/test size: {}/{}/{}'.format(len(train_set_exp), len(dev_set_exp), len(test_set_exp)))
print('imp: train/dev/test size: {}/{}/{}'.format(len(train_set_imp), len(dev_set_imp), len(test_set_imp)))
