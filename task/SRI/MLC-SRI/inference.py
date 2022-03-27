import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils import CustomDataset, load_json, write_json

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target", default='exp', required=False, type=str)
parser.add_argument("--cuda_num", default='1', required=False, type=str)
args = parser.parse_args()


prefix = 'sri_data'
model_prefix = './saved'

# define device
device = 'cuda:{}'.format(args.cuda_num) if torch.cuda.is_available() else 'cpu'
print(device)

# load normalized symptom
id2sym = {key: value for key, value in pd.read_csv('../../../dataset/symptom_norm.csv')['norm'].items()}

# load model
best_epoch = 41
model = torch.load(os.path.join(model_prefix, args.target, 'model_{}.pkl'.format(best_epoch)))
model.to(device)

# load test set
test = load_json(os.path.join(prefix, 'test_set_{}.json'.format(args.target)))
# test = load_json('sri_data/test_set.json')

MAX_LEN = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

test_set = CustomDataset(test, tokenizer, MAX_LEN)

TEST_BATCH_SIZE = 64

test_params = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 1
}

test_loader = DataLoader(test_set, **test_params)


def inference():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader), total=len(test_set) // TEST_BATCH_SIZE + 1):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            _targets = data['targets'].to(device, dtype=torch.float)
            _outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(_targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(_outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


outputs, targets = inference()
outputs = np.array(outputs) >= 0.5

# convert to true label
sids = []
for i in range(len(test)):
    if test[i][2] not in sids:
        sids.append(test[i][2])


final = {}
start = 0
end = 0

if args.target == 'exp':
    for i in range(len(sids)):
        sid = sids[i]
        labels, = np.where(outputs[i])
        pl = []
        for label in labels:
            pl.append(id2sym.get(label))
        final[str(sid)] = pl
else:
    for i in range(len(sids)):
        sid = sids[i]
        while end < len(test) and test[end][2] == sid:
            end += 1
        _, labels = np.where(outputs[start: end])
        pl = {}
        for label in labels:
            pl[id2sym[label // 3]] = str(int(label % 3))
        final[str(sid)] = pl
        start = end

write_json(final, 'mlc_{}_pred.json'.format(args.target))
