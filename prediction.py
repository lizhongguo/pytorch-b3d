import torch
from collections.abc import Iterable
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

    def __str__(self):
        _str_format = "%.2f\t"*len(self.labels)+'\n'
        _str = ' \t' + '\t'.join(l for l in self.labels) + '\n'
        for l in range(len(self.labels)):
            _str = _str + '%s\t' % self.labels[l] + _str_format % \
                tuple((i/self._matrix[l].sum()).item()
                      for i in self._matrix[l])
        return _str

model = 'i3d'
dataset = 'pev'
split_list = open('/home/lizhongguo/dataset/pev_split/val_split_3.txt')

id2label = dict()

for i, s in enumerate(split_list):
    s = s.split(' ')
    label = int(s[2])        
    id2label[i] = label


rgb = torch.load('%s_%s_%s_result.pt' % (dataset, model, 'rgb'))
flow = torch.load('%s_%s_%s_result.pt' % (dataset, model, 'flow'))

top1 = AverageMeter()
top2 = AverageMeter()
label_names = ['0', '1', '2', '3', '4', '5', '6']
confusion_matrix = MatrixMeter(label_names)

pred_result = rgb
pred_result.update(flow)

for i in pred_result:
    avg_pred = torch.stack(
        tuple(o for o in pred_result[i]), dim=0).mean(dim=0)
    target = id2label[i.item()]
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

print("Top1:%.2f Top2:%.2f" % (top1.avg*100, top2.avg*100))
print(confusion_matrix)






