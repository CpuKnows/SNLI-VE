import json
import os
import re

from scripts import config

for snli_file, ds in zip([os.path.join(config.DATA_DIR, 'snli_ve_train.jsonl'),
                          os.path.join(config.DATA_DIR, 'snli_ve_dev.jsonl'),
                          os.path.join(config.DATA_DIR, 'snli_ve_test.jsonl')],
                         ['train', 'dev', 'test']):
    dataset = []
    with open(snli_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    with open(os.path.join(config.FASTTEXT_DIR, f'fasttext_{ds}.txt'), 'w') as f:
        for sample in dataset:
            f.write('__label__{} {}\n'.format(sample['gold_label'], re.sub(r'\r\n|\n', r' ', sample['sentence2'])))
