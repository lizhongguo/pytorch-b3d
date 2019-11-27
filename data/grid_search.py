from thundersvm import *
from sklearn.datasets import *
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--split',  type=int, default=1)
parser.add_argument('--mode',  type=str, default='rgb')
parser.add_argument('--model',  type=str, default='i3d')
parser.add_argument('--fuse',  type=str, default='cbp')
args = parser.parse_args()

score = 0.
score_in_splits = [0.]*3

for gamma in [2**i/28 for i in [-2,-1,0,1,2,3,4,5,6]]:
    for C in [i/128 for i in range(1,129)]:
        split = args.split
        svm = SVC(kernel='rbf',gamma=gamma,C=C)
        training_set = load_svmlight_file('pev_split_%d_%s_%s_%strain_svm.txt' % (split, args.model, args.mode, \
            '' if args.fuse=='cbp' else args.fuse+'_'))
        svm.fit(training_set[0], training_set[1])
        validation_set = load_svmlight_file('pev_split_%d_%s_%s_%sval_svm.txt' % (split, args.model, args.mode, \
            '' if args.fuse=='cbp' else args.fuse+'_'))
        predict = svm.predict(validation_set[0])
        #score = svm.score(validation_set[0],validation_set[1])
        cm = confusion_matrix(validation_set[1],predict)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        score_in_splits[split-1] = np.mean([ cm[i][i] for i in range(7)])
        #print('gamma:%.4f C:%.4f score : %.4f' % (gamma, C, score_in_splits[split-1]))
        if score < sum(score_in_splits)/1:
            score = sum(score_in_splits)/1
            print('gamma:%.4f C:%.4f best score : %.4f' % (gamma, C, score))

#np.set_printoptions(precision=2)
#print(np.mean([ cm[i][i] for i in range(7)]))
#print([cm[i][i] for i in range(7)])
#print(cm)
