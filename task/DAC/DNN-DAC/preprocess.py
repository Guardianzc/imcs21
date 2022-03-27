import pandas as pd
import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


prefix = 'dataset'


def make_data(samples):
    pass

