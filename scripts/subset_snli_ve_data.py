import json
import os
import random

from scripts import config

dataset = []
with open(os.path.join(config.DATA_DIR, 'snli_ve_train.jsonl'), 'r') as f:
    for line in f:
        dataset.append(json.loads(line))

limit = 1000
with open(os.path.join(config.DATA_DIR, 'snli_ve_1k_train.jsonl'), 'w') as f:
    for idx in random.sample(range(len(dataset)), limit):
        json.dump(dataset[idx], f)
        f.write('\n')

limit = 10000
with open(os.path.join(config.DATA_DIR, 'snli_ve_10k_train.jsonl'), 'w') as f:
    for idx in random.sample(range(len(dataset)), limit):
        json.dump(dataset[idx], f)
        f.write('\n')

limit = 100000
with open(os.path.join(config.DATA_DIR, 'snli_ve_100k_train.jsonl'), 'w') as f:
    for idx in random.sample(range(len(dataset)), limit):
        json.dump(dataset[idx], f)
        f.write('\n')
