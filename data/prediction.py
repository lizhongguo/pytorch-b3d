import torch
from collections.abc import Iterable
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--split',  type=int, default=1)
parser.add_argument('--mode',  type=str, default='rgb')
parser.add_argument('--comment',  type=str, default='')
parser.add_argument('--fuse',  type=str, default='svm')
parser.add_argument('--view',  type=str, default='f')
parser.add_argument('--model',  type=str, default='i3d')

args = parser.parse_args()

y_true = []
y_score = []

val = open('pev_split_%d_%s_%s_%s_svm.txt' %(args.split, args.model, args.mode, 'val'))
predict = open('pev_split_%d_%s_%s_%s_svm.txt.result' %(args.split, args.model, args.mode, 'val'))

for v,p in zip(val,predict):
    y_true.append(int(p))
    y_score.append(int(v.split(' ', 1)[0]))

label_names = ['Pit', 'Att', 'Pas', 'Rec', 'Pos', 'Neg', 'Ges']
print(label_names)
y_true = np.array(y_true)
y_score = np.array(y_score)

cm = confusion_matrix(y_true,y_score)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.save('pev_split_%d_%s_%s_%s_%s_%s.npy' %(args.split, args.model,args.mode, args.view, args.fuse, \
     args.comment), [ cm[i][i] for i in range(7)])

np.set_printoptions(precision=2)
print(np.mean([ cm[i][i] for i in range(7)]))
print([cm[i][i] for i in range(7)])
print(cm)

