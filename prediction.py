import torch
from collections.abc import Iterable
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--split',  type=int, default=1)
parser.add_argument('--mode',  type=str, default='rgb')

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def update_confusion_matrix(matrix_meter, output, target):
    """Computes the precision@k for the specified values of k"""

    #output = output.view((-1, 7, 3, 3))
    #output = torch.einsum('n%s->n%s' % ('abc', 'a'), output)
    #target = target[:, 0]

    maxk = 1
    _, pred = output.topk(maxk, 1, True, True)
    matrix_meter.update(pred.t()[0], target)


class MatrixMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, labels):
        self.labels = labels
        self._matrix = torch.zeros(
            (len(labels), len(labels)), dtype=torch.float)

    def update(self, output, target):
        # pdb.set_trace()

        if isinstance(output, Iterable) and isinstance(target, Iterable):
            for o, t in zip(output, target):
                self._matrix[t][o] = self._matrix[t][o] + 1.
        else:
            self._matrix[target][output] = self._matrix[target][output] + 1.

    @property
    def _data(self):
        data = torch.zeros(
            (len(self.labels), len(self.labels)), dtype=torch.float)

        for l in range(len(self.labels)):
            data[l] = self._matrix[l]/self._matrix[l].sum()
        return data

    @property
    def acc(self):
        data = torch.zeros(
            len(self.labels), dtype=torch.float)

        for l in range(len(self.labels)):
            data[l] = self._matrix[l][l]/self._matrix[l].sum()
        return data.mean().item()

    def __str__(self):
        _str_format = "%.2f\t"*len(self.labels)+'\n'
        _str = ' \t' + '\t'.join(l for l in self.labels) + '\n'
        for l in range(len(self.labels)):
            _str = _str + '%s\t' % self.labels[l] + _str_format % \
                tuple((i/self._matrix[l].sum()).item()
                      for i in self._matrix[l])
        return _str

split_idx = args.split
model = 'i3d'
dataset = 'pev'
split_list = open('/home/lizhongguo/dataset/pev_split/val_split_%d.txt' % split_idx)
view = 's'
id2label = dict()

for i, s in enumerate(split_list):
    s = s.split(' ')
    label = int(s[2])        
    id2label[i] = label


rgb = torch.load('%s_split_%d_%s_%s_%s_result.pt' % (dataset, split_idx , model, args.mode, 'f'))
flow = torch.load('%s_split_%d_%s_%s_%s_result.pt' % (dataset,split_idx , model, args.mode, 's'))

top1 = AverageMeter()
top2 = AverageMeter()
label_names = ['0', '1', '2', '3', '4', '5', '6']
confusion_matrix = MatrixMeter(label_names)

rgb = { k.item():v for k,v in rgb.items() }
flow = { k.item():v for k,v in flow.items() }

pred_result = dict()
for k,v in rgb.items():
    pred_result[k] = v
    pred_result[k].extend(flow[k])

y_true = []
y_score = []

for i in pred_result:
    avg_pred = torch.stack(
        tuple(o for o in pred_result[i]), dim=0).mean(dim=0)
    target = id2label[i]
    _, prediction = avg_pred.topk(2)
    prediction = prediction.tolist()
    if target == prediction[0]:
        top1.update(1., n=1)
    else:
        top1.update(0., n=1)

    if target in prediction:
        top2.update(1., n=1)
    else:
        top2.update(0., n=1)

    confusion_matrix.update(prediction[0], target)

    y_true.append(target)
    y_score.append(avg_pred.cpu().tolist())

label_names = ['Pit', 'Att', 'Pas', 'Rec', 'Pos', 'Neg', 'Ges']
print(label_names)
y_true = np.array(y_true)
y_score = np.array(y_score)
for l in range(7): #for each label compute corresponding auc score
    gt = np.copy(y_true)
    gt[y_true == l] = 1
    gt[y_true != l] = 0
    score = np.copy(y_score)
    score = score[:,l]
    print(l,roc_auc_score(gt, score))

print("Top1:%.2f Top2:%.2f" % (confusion_matrix.acc*100, top2.avg*100))
print(confusion_matrix)






