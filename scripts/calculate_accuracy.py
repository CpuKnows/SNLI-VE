import argparse
import json
import numpy as np


parser = argparse.ArgumentParser(description='Calculate accuracy of predictions')
parser.add_argument('true_label_path', type=str, help='path to true labels')
parser.add_argument('pred_label_path', type=str, help='path to predicted labels')
args = parser.parse_args()

true_labels = []
with open(args.true_label_path, 'r') as f:
    for line in f:
        true_labels.append(json.loads(line)['gold_label'])

pred_labels = []
with open(args.pred_label_path, 'r') as f:
    for line in f:
        pred_labels.append(json.loads(line)['prediction'])

assert len(true_labels) == len(pred_labels), "Files must have the same number of labels"
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

entail_idx = (true_labels == 'entailment')
neutral_idx = (true_labels == 'neutral')
contradict_idx = (true_labels == 'contradiction')

print('Overall accuracy = {:.2%}'.format(np.sum(true_labels == pred_labels) / len(true_labels)))
print('Entailed accuracy = {:.2%}'.format(
    np.sum(true_labels[entail_idx] == pred_labels[entail_idx]) / len(true_labels[entail_idx])))
print('Neutral accuracy = {:.2%}'.format(
    np.sum(true_labels[neutral_idx] == pred_labels[neutral_idx]) / len(true_labels[neutral_idx])))
print('Contradict accuracy = {:.2%}'.format(
    np.sum(true_labels[contradict_idx] == pred_labels[contradict_idx]) / len(true_labels[contradict_idx])))
