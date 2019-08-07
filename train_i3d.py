from multi_label_loss import MLL
from multi_label_loss import MLLSampler

from tensorboardX import SummaryWriter
from spatial_transform import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from target_transform import ClassLabel
from temporal_transform import TemporalRandomCrop
from pev import PEV
from charades_dataset import Charades as Dataset
from pytorch_i3d import InceptionI3d
import numpy as np
import videotransforms
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

from confusion_matrix_figure import draw_confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()
top_acc = 0


def run(init_lr=0.1, max_steps=320, mode='rgb', batch_size=32, save_model=''):
    logger = SummaryWriter()

    # setup dataset
    train_transforms = Compose([MultiScaleRandomCrop([1.0, 0.8, 0.64], 224),
                                RandomHorizontalFlip(),
                                ToTensor(1.0),
                                Normalize([0, 0, 0], [1, 1, 1])
                                ])

    test_transforms = Compose([MultiScaleRandomCrop([1.0], 224),
                               ToTensor(1.0),
                               Normalize([0, 0, 0], [1, 1, 1])
                               ])
    temporal_transforms = TemporalRandomCrop(64)
    target_transforms = ClassLabel()

    #dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataset = PEV('/dataset/pev_frames',
                  '/dataset/pev_split/train_split.txt',
                  'training',
                  spatial_transform=test_transforms,
                  temporal_transform=temporal_transforms,
                  target_transform=target_transforms,
                  sample_duration=64)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = PEV(
        '/dataset/pev_frames',
        '/dataset/pev_split/val_split.txt',
        'validation',
        1,
        spatial_transform=train_transforms,
        temporal_transform=temporal_transforms,
        target_transform=target_transforms,
        sample_duration=64)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=36, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(num_classes=7, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(num_classes=7,
                           in_channels=3, dropout_keep_prob=0.5)
        i3d.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                             if k.find('logits') < 0}, strict=False)
    # i3d.replace_logits(157)
    # i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])

    steps = 0
    # criterion = MLL(dataset.multi_label_shape)    # train it
    criterion = nn.CrossEntropyLoss()
    count = 0

    #sampler = MLLSampler(dataloaders['train'],dataset.multi_label_shape)
    em_steps = 40

    while steps < max_steps:
        count = train(
            i3d, dataloaders['train'], criterion, optimizer, lr_sched, count, logger)
        val(i3d, dataloaders['val'], criterion, steps, logger)

        if (steps+1) % em_steps == 0:
            # i3d.load_state_dict(torch.load('pev_i3d_best.pt'))
            # sampler.sample_hidden_state(i3d)
            optimizer = optim.SGD(i3d.parameters(), lr=init_lr,
                                  momentum=0.9, weight_decay=0.0000001)
            lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])

        steps = steps + 1

    logger.close()


def train(model, dataloader, criterion, optimizer, lr_sched, count, logger=None):
    model.train(True)

    for data in dataloader:
        optimizer.zero_grad()
        inputs, labels, _ = data
        inputs = inputs.cuda()
        labels = labels.cuda(non_blocking=True)

        output = model(inputs)

        # labels follow the shape of multi label shape
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        top1, top2 = accuracy(output, labels, (1, 2))
        count = count + 1

        logger.add_scalar('train/loss', loss.item(), count)
        logger.add_scalar('train/top1', top1, count)

        print("Iteration %d Loss %.4f Top1:%.2f Top2:%.2f" %
              (count, loss.item(), top1, top2))

    # lr_sched.step()
    return count


def val(model, dataloader, criterion, epoch, logger=None):
    model.train(False)
    top1 = AverageMeter()
    top2 = AverageMeter()
    val_loss = AverageMeter()
    label_names = ['0','1','2','3','4','5','6']
    confusion_matrix = MatrixMeter(label_names)
    global top_acc
    with torch.no_grad():
        for it, data in enumerate(dataloader):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda(non_blocking=True)
            output = model(inputs)

            loss = criterion(output, labels)
            val_loss.update(loss.item(), labels.size(0))

            a1, a2 = accuracy(output, labels, (1, 2))
            update_confusion_matrix(confusion_matrix, output, labels)

            top1.update(a1, labels.size(0))
            top2.update(a2, labels.size(0))

        if top1.avg > top_acc:
            top_acc = top1.avg
            torch.save(model.state_dict(),
                       'pev_i3d_best.pt')

        logger.add_scalar('val/top1', top1.avg, epoch)
        logger.add_scalar('val/loss', val_loss.avg, epoch)
        logger.add_figure('val/confusion_matrix',
                          draw_confusion_matrix(confusion_matrix._data,label_names),epoch)

        print("Top1:%.2f Top2:%.2f" % (top1.avg, top2.avg))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #output = output.view((-1, 7, 2, 3, 2, 2, 2, 4))
    # output = output.view((-1, 7, 7))

    #output = torch.einsum('n%s->n%s' % ('abcdefg', 'a'), output)
    #output = torch.einsum('n%s->n%s' % ('ab', 'a'), output)
    #target = target[:, 0]

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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

    #output = output.view((-1, 7, 2, 3, 2, 2, 2, 4))
    #output = torch.einsum('n%s->n%s' % ('abcdefg', 'a'), output)
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
        for o, t in zip(output, target):
            self._matrix[t][o] = self._matrix[t][o] + 1

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


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, batch_size=15, save_model=args.save_model)
