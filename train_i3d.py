import pdb
from tqdm import tqdm
from confusion_matrix_figure import draw_confusion_matrix
from multi_label_loss import MLL
from multi_label_loss import MLLSampler
from collections.abc import Iterable
from R2Plus1D import R2Plus1DClassifier
from W3D import W3D
from torch.utils.tensorboard import SummaryWriter
from spatial_transform import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from target_transform import ClassLabel, VideoID
from temporal_transform import TemporalRandomCrop, TemporalCenterCrop, TemporalBeginCrop
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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--save_model', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--model', type=str, choices=['i3d', 'r2plus1d', 'w3d'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--resume', type=str, default=None)

from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_apply

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default='O2')
parser.add_argument('--keep-batchnorm-fp32', action='store_true')
parser.add_argument('--loss-scale', type=float, default=128.)
parser.add_argument('--apex', action='store_true')


args = parser.parse_args()
top_acc = 0

if args.apex:
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    #args.gpu = 0
    args.world_size = 1

    if args.distributed:
        #args.gpu = args.local_rank
        #torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    # for tuning gpu to speed up training
    torch.backends.cudnn.benchmark = True


def model_builder():
    # setup the model
    if args.model == 'i3d':
        if args.mode == 'flow':
            model = InceptionI3d(num_classes=7, in_channels=2)
            model.load_state_dict(torch.load('models/flow_imagenet.pt'))
        else:
            model = InceptionI3d(num_classes=7,
                                 in_channels=3, dropout_keep_prob=0.5, Pooling='Max')
            model.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                   if k.find('logits') < 0}, strict=False)
    elif args.model == 'r2plus1d':
        model = R2Plus1DClassifier(num_classes=7)
    elif args.model == 'w3d':
        model = W3D(num_classes=7)
        # model.load_state_dict(torch.load('pev_i3d_best.pt'))

    if args.sync_bn and args.apex:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    if args.resume is not None:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(
                    args.resume, map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint)
                print("=> loaded checkpoint '{}' "
                      .format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    model = model.cuda()

    #lr = args.init_lr * args.batch_size
    lr = args.lr * args.batch_size * args.world_size / 64.
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.0000001)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])
    if args.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                        opt_level=args.opt_level,
                                        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                        loss_scale=args.loss_scale
                                        )

        if args.distributed:
            model = DDP(model, delay_allreduce=True)
    else:
        model = nn.DataParallel(model)

    return model, optimizer


def run(max_steps=80, mode='rgb', batch_size=32, save_model=''):
    logger = SummaryWriter()

    if args.model in ('i3d', ):
        scale_size = 224
    elif args.model in ('r2plus1d', 'w3d'):
        scale_size = 112
    else:
        raise Exception('Model %s not implemented' % args.model)

    # setup dataset
    train_transforms = Compose([MultiScaleRandomCrop([1.0, 0.9, 0.81], scale_size),
                                RandomHorizontalFlip(),
                                ToTensor(1.0),
                                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

    test_transforms = Compose([MultiScaleRandomCrop([1.0], scale_size),
                               ToTensor(1.0),
                               Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])
    temporal_transforms = TemporalRandomCrop(64)
    target_transforms = ClassLabel()

    #dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataset = PEV('/home/lizhongguo/dataset/pev_frames',
                  '/home/lizhongguo/dataset/pev_split/train_split_3.txt',
                  'training',
                  n_samples_for_each_video=6,
                  spatial_transform=train_transforms,
                  temporal_transform=temporal_transforms,
                  target_transform=target_transforms,
                  sample_duration=64)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    val_dataset = PEV(
        '/home/lizhongguo/dataset/pev_frames',
        '/home/lizhongguo/dataset/pev_split/val_split_3.txt',
        'validation',
        6,
        spatial_transform=test_transforms,
        temporal_transform=temporal_transforms,
        target_transform=target_transforms,
        sample_duration=64)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}

    model, optimizer = model_builder()

    steps = 0
    # criterion = MLL(dataset.multi_label_shape)    # train it
    criterion = nn.CrossEntropyLoss().cuda()
    count = 0

    while steps < max_steps:
        count = train(
            model, dataloaders['train'], criterion, optimizer, None, count, logger)
        val(model, dataloaders['val'], criterion, steps, logger)

        # i3d.load_state_dict(torch.load('pev_i3d_best.pt'))
        dataset.undersample(dataset.root_path, dataset.raw_data, dataset.subset,
                            dataset.min_class_len, dataset.n_samples_for_each_video, dataset.sample_duration)

        steps = steps + 1

    logger.close()


def evaluate(init_lr=0.1, max_steps=320, mode='rgb', batch_size=20, save_model=''):

    logger = SummaryWriter()

    # setup dataset

    test_transforms = Compose([MultiScaleRandomCrop([1.0], 224),
                               ToTensor(1.0),
                               Normalize([0, 0, 0], [1, 1, 1])
                               ])
    temporal_transforms = TemporalCenterCrop(64)
    target_transforms = VideoID()

    val_dataset = PEV(
        '/home/lizhongguo/dataset/pev_frames',
        '/home/lizhongguo/dataset/pev_split/val_split_3.txt',
        'evaluation',
        6,
        spatial_transform=test_transforms,
        temporal_transform=temporal_transforms,
        target_transform=target_transforms,
        sample_duration=64)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # setup the model
    model, _ = model_builder()
    model.train(False)
    top1 = AverageMeter()
    top2 = AverageMeter()
    label_names = ['0', '1', '2', '3', '4', '5', '6']
    confusion_matrix = MatrixMeter(label_names)

    pred_result = dict()

    with torch.no_grad():
        for data in tqdm(val_dataloader):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda(non_blocking=True)
            output = model(inputs)
            output = F.softmax(output, dim=1)

            for o, i in zip(output, labels):
                if i not in pred_result:
                    pred_result[i] = []
                pred_result[i].append(o)

        for i in pred_result:
            avg_pred = torch.stack(
                tuple(o for o in pred_result[i]), dim=0).mean(dim=0)
            target = val_dataset.id2label[i.item()]
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

    logger.add_scalar('val/top1', 100*top1.avg, 0)
    logger.add_scalar('val/top2', 100*top2.avg, 0)
    logger.add_figure('val/confusion',
                      draw_confusion_matrix(confusion_matrix._data, label_names), 0, close=False)

    print("Top1:%.2f Top2:%.2f" % (top1.avg*100, top2.avg*100))
    print(confusion_matrix)

    logger.close()


def train(model, dataloader, criterion, optimizer, lr_sched, count, logger=None):
    model.train(True)

    for data in tqdm(dataloader):
        inputs, labels, _ = data

        inputs = inputs.cuda()
        labels = labels.cuda(non_blocking=True)

        output = model(inputs)

        optimizer.zero_grad()

        # labels follow the shape of multi label shape
        loss = criterion(output, labels)

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        top1, top2 = accuracy(output, labels, (1, 2))
        count = count + 1

        logger.add_scalar('train/loss', loss.item(), count)
        logger.add_scalar('train/top1', top1, count)

        # print("Iteration %d Loss %.4f Top1:%.2f Top2:%.2f" %
        #      (count, loss.item(), top1, top2))

    # lr_sched.step()
    return count


def val(model, dataloader, criterion, epoch, logger=None):
    model.train(False)
    top1 = AverageMeter()
    top2 = AverageMeter()
    val_loss = AverageMeter()
    label_names = ['0', '1', '2', '3', '4', '5', '6']
    confusion_matrix = MatrixMeter(label_names)
    global top_acc
    with torch.no_grad():
        for data in tqdm(dataloader):
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
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(),
                           '%s_%s_best.pt' % ('pev', args.model))
            else:
                torch.save(model.state_dict(),
                           '%s_%s_best.pt' % ('pev', args.model))

        logger.add_scalar('val/top1', top1.avg, epoch)
        logger.add_scalar('val/top2', top2.avg, epoch)
        logger.add_scalar('val/loss', val_loss.avg, epoch)
        logger.add_figure('val/confusion',
                          draw_confusion_matrix(confusion_matrix._data, label_names), epoch, close=False)

        print("Top1:%.2f Top2:%.2f" % (top1.avg, top2.avg))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #output = output.view((-1, 7, 2, 3, 2, 2, 2, 4))
    #output = output.view((-1, 7, 3, 3))

    #output = torch.einsum('n%s->n%s' % ('abcdefg', 'a'), output)
    #output = torch.einsum('n%s->n%s' % ('abc', 'a'), output)
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


if __name__ == '__main__':
    # need to add argparse
    if args.eval:
        evaluate(batch_size=10)
    else:
        run(max_steps=80, mode=args.mode,
            batch_size=args.batch_size, save_model=args.save_model)
