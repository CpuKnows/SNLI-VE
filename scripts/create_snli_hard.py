import json
import os
import numpy as np

from scripts import config


for snli_file, split in zip([os.path.join(config.DATA_DIR, 'snli_ve_train.jsonl'),
                             os.path.join(config.DATA_DIR, 'snli_ve_dev.jsonl'),
                             os.path.join(config.DATA_DIR, 'snli_ve_test.jsonl')],
                            ['train', 'dev', 'test']):
    dataset = []
    with open(snli_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    y_true = []
    with open(os.path.join(config.FASTTEXT_DIR, f'fasttext_{split}.txt'), 'r') as f:
        for line in f:
            y_true.append(line.strip().split(' ')[0].split('__')[-1])

    y_pred = []
    with open(os.path.join(config.FASTTEXT_DIR, f'prediction_{split}.txt'), 'r') as f:
        for line in f:
            y_pred.append(line.strip().split('__')[-1])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    hard_idxs = np.where(y_true != y_pred)[0]

    print(f'Processing {split} dataset:\n' +
          f'{len(dataset)} samples in original dataset.\n' +
          f'{len(hard_idxs)} samples in hard dataset. ({len(hard_idxs) / len(dataset)}%) \n\n')

    hard_dataset = []
    for idx in hard_idxs:
        hard_dataset.append(dataset[idx])

    with open(os.path.join(config.DATA_DIR, f'snli_ve_hard_{split}.jsonl'), 'w') as f:
        for sample in hard_dataset:
            json.dump(sample, f)
            f.write('\n')
