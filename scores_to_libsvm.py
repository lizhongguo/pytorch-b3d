import torch
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
parser = argparse.ArgumentParser()
parser.add_argument('--split',  type=int, default=1)
parser.add_argument('--mode',  type=str, default='rgb')
parser.add_argument('--model',  type=str, default='i3d')
parser.add_argument('--fuse',  type=str, default='cbp')
args = parser.parse_args()

for subset in ['train', 'val']:
    split_list = open('/home/lizhongguo/dataset/pev_split/%s_split_%d.txt'
                      % (subset, args.split))

    id2label = dict()
    for s in split_list:
        s = s.split(' ')
        label = int(s[2])
        id2label[int(s[0][:-2])] = label

    first_view = {}
    data = torch.load('pev_split_%d_%s_%s_%s_%s%s_scores.pt' %
                      (args.split, args.model, args.mode, 'fs', '' if args.fuse=='cbp' else args.fuse+'_', subset))
    split_txt = open('pev_split_%d_%s_%s_%s%s_svm.txt' %
                     (args.split, args.model, args.mode, '' if args.fuse=='cbp' else args.fuse+'_', subset), 'w')
    y_true = []
    y_score_1 = []

    for l in data:
        label = id2label[l[1].item()]
        feature = [e for e in l[0]]

        y_true.append(label)
        y_score_1.append(l[0][:7].argmax(axis=0))
        formattxt = ('%d %s\n') % (label, ' '.join(
            ['%d:%.4f' % (i+1, e) for i, e in enumerate(feature)]))
        split_txt.write(formattxt)
    print(accuracy_score(y_true,y_score_1))
