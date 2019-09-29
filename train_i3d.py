from BPTSN import BPTSN
from tsn import TSN
import pdb
from tqdm import tqdm
from confusion_matrix_figure import draw_confusion_matrix
from multi_label_loss import MLL
from multi_label_loss import MLLSampler
from collections.abc import Iterable
from R2Plus1D import R2Plus1DClassifier
from W3D import W3D
from BPI3D import BPI3D
from BPC3D import BPC3D
from DI3D import DI3D

from torch.utils.tensorboard import SummaryWriter
from spatial_transform import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from target_transform import ClassLabel, VideoID
from temporal_transform import TemporalRandomCrop, TemporalCenterCrop, TemporalBeginCrop, LoopPadding, RepeatPadding
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

#from mi3d import MInceptionI3d

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--save_model', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--model', type=str,
                    choices=['i3d', 'r2plus1d', 'w3d', 'tsn', 'bptsn', 'bpi3d', 'bpc3d', 'di3d'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sample_freq', type=int, default=1)
parser.add_argument('--sample_step', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=6,
                    help='num of samples for each video')
parser.add_argument('--clip_len', type=int, default=32)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--print_loss', action='store_true')
parser.add_argument('--view', type=str, choices=['s', 'f', 'fs'])

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str, default='O2')
parser.add_argument('--keep-batchnorm-fp32', action='store_true')
parser.add_argument('--loss-scale', type=str, default="dynamic")
parser.add_argument('--apex', action='store_true')
parser.add_argument('--split_idx', type=int, default=1)


args = parser.parse_args()
top_acc = 0.

split_idx = args.split_idx

if args.apex:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_apply
else:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

#args.gpu = 0
args.world_size = 1

if args.distributed:
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
# for tuning gpu to speed up training
torch.backends.cudnn.benchmark = True

#print('rank %d started' % torch.distributed.get_rank())

args.main_rank = (args.distributed and torch.distributed.get_rank()
                  == 0) or (not args.distributed)
data_root = '/home/lizhongguo/dataset/pev_of'


def model_builder():
    global top_acc
    # setup the model
    if args.model == 'i3d':
        if args.mode == 'flow':
            model = InceptionI3d(num_classes=7, in_channels=2)
            model.load_state_dict({k: v for k, v in torch.load('models/flow_imagenet.pt').items()
                                   if k.find('logits') < 0}, strict=False)
        elif args.mode == 'rgb':
            model = InceptionI3d(num_classes=7,
                                 in_channels=3, dropout_keep_prob=0.5)
            model.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                   if k.find('logits') < 0}, strict=False)
        elif args.mode == 'rgb+flow':
            model = MInceptionI3d(num_classes=7,
                                  in_channels=5, dropout_keep_prob=0.5)
            model.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                   if k.find('logits') < 0}, strict=False)
    elif args.model == 'bpi3d':
        if args.mode == 'flow':
            model = BPI3D(num_classes=7, in_channels=2)
            model.backbone.load_state_dict(torch.load(
                'models/flow_imagenet.pt'), strict=False)
        elif args.mode == 'rgb':
            model = BPI3D(num_classes=7,
                          in_channels=3, dropout_keep_prob=0.5)
            model.backbone.load_state_dict(torch.load(
                'models/rgb_imagenet.pt'), strict=False)
            # model.backbone.load_state_dict({k: v for k, v in torch.load('pev_split_%d_i3d_rgb_best.pt' % split_idx)['state_dict'].items()
            #                                if k.find('logits') < 0}, strict=False)

    elif args.model == 'r2plus1d':
        model = R2Plus1DClassifier(num_classes=7)
    elif args.model == 'w3d':
        model = W3D(num_classes=7)
        # model.load_state_dict(torch.load('pev_i3d_best.pt'))

    elif args.model == 'tsn':
        if args.mode == 'rgb':
            model = TSN(num_classes=7, in_channels=3)
        else:
            model = TSN(num_classes=7, in_channels=2, transform_input=False)

    elif args.model == 'bptsn':
        if args.mode == 'rgb':
            model = BPTSN(num_classes=7, in_channels=3)
        else:
            model = BPTSN(num_classes=7, in_channels=2, transform_input=False)

    elif args.model == 'bpc3d':
        if args.mode == 'rgb':
            model = BPC3D(num_classes=7, in_channels=3)
        else:
            model = BPC3D(num_classes=7, in_channels=2)

    elif args.model == 'di3d':
        model = DI3D(num_classes=7)

        model.backbone_flow.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                            if k.find('logits') < 0}, strict=False)
        ''''
        model.backbone_flow.load_state_dict(torch.load(
            '%s_split_%d_%s_%s_%s_%s.pt' % (
                'pev', split_idx, 'i3d', 'flow', 'best', args.view),
            map_location=lambda storage, loc: storage)['state_dict'])
        model.backbone_rgb.load_state_dict(torch.load(
            '%s_split_%d_%s_%s_%s_%s.pt' % (
                'pev', split_idx, 'i3d', 'rgb', 'best', args.view),
            map_location=lambda storage, loc: storage)['state_dict'])

        '''
        model.backbone_rgb.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                            if k.find('logits') < 0}, strict=False)
        '''
        model.backbone_s.load_state_dict(torch.load(
            '%s_split_%d_%s_%s_%s_%s.pt' % (
                'pev', split_idx, 'i3d', args.mode, 'best', 's'),
            map_location=lambda storage, loc: storage)['state_dict'])
        '''
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
                model.load_state_dict(checkpoint['state_dict'])
                top_acc = checkpoint['top_acc']
                print("=> loaded checkpoint '{}' '{}' "
                      .format(args.resume, top_acc))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    model = model.cuda()

    if args.apex:
        lr = args.lr * args.batch_size * args.world_size / 64.
    else:
        lr = args.lr * args.batch_size / 56.

    if args.model == 'bptsn':
        bp_params = model.bp.parameters()
        base_params = model.backbone.parameters()
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.*lr},
            {'params': bp_params, 'lr': lr}
        ], lr=lr,
            momentum=0.9, weight_decay=0.0000001)

    elif args.model == 'di3d':
        optimizer = optim.SGD([
            {'params': model.fc.parameters(), 'lr': lr},
            {'params': model.backbone_flow.parameters(), 'lr': lr},
            {'params': model.backbone_rgb.parameters(), 'lr': lr},
            #{'params': model.weight, 'lr': lr}
        ], lr=lr,
            momentum=0.9, weight_decay=0.0000001)

    else:
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
    elif args.distributed:
        model = DDP(model)
    else:
        model = nn.DataParallel(model)

    return model, optimizer


def run(max_steps=80, mode='rgb', batch_size=32, save_model=''):
    if args.main_rank:
        logger = SummaryWriter(comment='log_%s_%s_%s_split_%d' %
                               (args.model, args.mode, args.view, args.split_idx))
    else:
        logger = None

    if args.model == 'i3d' or args.model == 'bpi3d' or args.model == 'di3d':
        scale_size = 224
    elif args.model == 'r2plus1d' or args.model == 'w3d' or args.model == 'bpc3d':
        scale_size = 112
    elif args.model == 'tsn' or args.model == 'bptsn':
        scale_size = 224 #299
    else:
        raise Exception('Model %s not implemented' % args.model)

    if args.model == 'i3d' or args.model == 'r2plus1d' or args.model == 'bpi3d' or args.model == 'bpc3d' or args.model == 'di3d':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.model == 'tsn' or args.model == 'bptsn':
        if args.mode == 'rgb':
            mean = [0., 0., 0.]
            std = [1., 1., 1.]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
    else:
        raise Exception('Model %s not implemented' % args.model)

    # setup dataset
    train_transforms = Compose([MultiScaleRandomCrop([1.0], scale_size),
                                RandomHorizontalFlip(),
                                ToTensor(255.0),
                                Normalize(mean, std)
                                ])

    test_transforms = Compose([CenterCrop(scale_size),
                               ToTensor(255.0),
                               Normalize(mean, std)
                               ])

    clip_len = args.clip_len // args.sample_step
    temporal_transforms = Compose([TemporalRandomCrop(clip_len),
                                   RepeatPadding(clip_len)])
    target_transforms = ClassLabel()

    #dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    dataset = PEV(data_root,
                  '/home/lizhongguo/dataset/pev_split/train_split_%d.txt' % split_idx,
                  'training',
                  n_samples_for_each_video=args.n_samples,
                  spatial_transform=train_transforms,
                  temporal_transform=temporal_transforms,
                  target_transform=target_transforms,
                  sample_duration=clip_len, sample_freq=args.sample_freq,
                  mode=args.mode, sample_step=args.sample_step, view=args.view)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    if args.model == 'tsn' or args.model == 'bptsn':
        dataset.random_select = True
        dataset.dense_select_length = 5 if args.mode == 'flow' else 1

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=2, pin_memory=True, sampler=sampler, drop_last=False)

    val_temporal_transforms = Compose([TemporalBeginCrop(clip_len),
                                       RepeatPadding(clip_len)])
    val_dataset = PEV(
        data_root,
        '/home/lizhongguo/dataset/pev_split/val_split_%d.txt' % split_idx,
        'evaluation',
        args.n_samples,
        spatial_transform=test_transforms,
        temporal_transform=val_temporal_transforms,
        target_transform=VideoID(),
        sample_duration=clip_len, sample_freq=args.sample_freq, mode=args.mode, sample_step=args.sample_step, view=args.view)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        val_sampler = None

    if args.model == 'tsn' or args.model == 'bptsn':
        val_dataset.random_select = True
        val_dataset.dense_select_length = 5 if args.mode == 'flow' else 1


    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=(val_sampler is None), num_workers=2, pin_memory=True, sampler=val_sampler, drop_last=False)

    dataloaders = {'train': dataloader, 'val': val_dataloader}

    model, optimizer = model_builder()

    steps = 0
    # criterion = MLL(dataset.multi_label_shape)    # train it
    if args.model == 'bptsn':
        criterion = nn.MultiMarginLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    count = 0

    while steps < max_steps:
        count = train(
            model, dataloaders['train'], criterion, optimizer, None, count, logger)
        val(model, dataloaders['val'], criterion, steps, logger)

        # i3d.load_state_dict(torch.load('pev_i3d_best.pt'))
        dataset.undersample(dataset.root_path, dataset.raw_data, dataset.subset,
                            dataset.min_class_len, dataset.n_samples_for_each_video,
                            dataset.sample_duration, dataset.sample_freq, dataset.sample_step)

        steps = steps + 1

    if logger is not None:
        logger.close()


def evaluate(init_lr=0.1, max_steps=320, mode='rgb', batch_size=20, save_model=''):

    logger = SummaryWriter()

    if args.model == 'i3d' or args.model == 'bpi3d' or args.model == 'di3d':
        scale_size = 224
    elif args.model == 'r2plus1d' or args.model == 'w3d' or args.model == 'bpc3d':
        scale_size = 112
    elif args.model == 'tsn' or args.model == 'bptsn':
        scale_size = 224 #299
    else:
        raise Exception('Model %s not implemented' % args.model)

    if args.model == 'i3d' or args.model == 'r2plus1d' or args.model == 'bpi3d' or args.model == 'bpc3d' or args.model == 'di3d':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.model == 'tsn' or args.model == 'bptsn':
        if args.mode == 'rgb':
            mean = [0., 0., 0.]
            std = [1., 1., 1.]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

    else:
        raise Exception('Model %s not implemented' % args.model)

    test_transforms = Compose([CenterCrop(scale_size),
                               ToTensor(255.0),
                               Normalize(mean, std)
                               ])
    clip_len = args.clip_len // args.sample_step

    temporal_transforms = Compose(
        [TemporalBeginCrop(clip_len), RepeatPadding(clip_len)])
    target_transforms = VideoID()
    val_dataset = PEV(
        data_root,
        '/home/lizhongguo/dataset/pev_split/val_split_%d.txt' % split_idx,
        'evaluation',
        args.n_samples,
        spatial_transform=test_transforms,
        temporal_transform=temporal_transforms,
        target_transform=target_transforms,
        sample_duration=clip_len, sample_freq=args.sample_freq, mode=args.mode, sample_step=args.sample_step, view=args.view)
    #val_dataset.random_select = True

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # setup the model
    model, _ = model_builder()
    model.train(False)

    if args.visualize:
        model.add_logger(logger, 'Conv3d_1a_7x7')

    top1 = AverageMeter()
    top2 = AverageMeter()
    label_names = ['Pit', 'Att', 'Pas', 'Rec', 'Pos', 'Neg', 'Ges']
    confusion_matrix = MatrixMeter(label_names)

    pred_result = dict()
    with torch.no_grad():
        for data in tqdm(val_dataloader):
            if args.model == 'di3d':
                input_f, input_s, labels, _ = data
                input_f, input_s = input_f.cuda(), input_s.cuda()
                labels = labels.cuda(non_blocking=True)
                output = model(input_f, input_s)

            else:
                inputs, labels, _ = data
                inputs = inputs.cuda()
                labels = labels.cuda(non_blocking=True)
                output = model(inputs)

            output = F.softmax(output, dim=1)

            for o, i in zip(output, labels):
                if i not in pred_result:
                    pred_result[i] = []
                pred_result[i].append(o)

        torch.save(pred_result, '%s_split_%d_%s_%s_%s_result.pt' %
                   ('pev', args.split_idx, args.model, args.mode, args.view))

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

    print("Top1:%.2f Top2:%.2f" % (confusion_matrix.acc*100, top2.avg*100))
    print(confusion_matrix)

    logger.close()


def train(model, dataloader, criterion, optimizer, lr_sched, count, logger=None):
    model.train(True)

    for data in tqdm(dataloader):
        if args.model == 'di3d':
            input_f, input_s, labels, _ = data
            input_f, input_s = input_f.cuda(), input_s.cuda()
            labels = labels.cuda(non_blocking=True)
            output = model(input_f, input_s)
            loss = criterion(output, labels)

        else:
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda(non_blocking=True)
            output = model(inputs)
            # labels follow the shape of multi label shape
            loss = criterion(output, labels)

        optimizer.zero_grad()

        if args.apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        top1, top2 = accuracy(output, labels, (1, 2))
        count = count + 1

        if logger is not None:
            logger.add_scalar('train/loss', loss.item(), count)
            logger.add_scalar('train/top1', top1, count)

        if args.print_loss:
            print("Iteration %d Loss %.4f Top1:%.2f Top2:%.2f" %
                  (count, loss.item(), top1, top2))

    # lr_sched.step()
    return count


def val(model, dataloader, criterion, epoch, logger=None):
    model.train(False)
    top1 = AverageMeter()
    top2 = AverageMeter()
    val_loss = AverageMeter()
    label_names = ['Pit', 'Att', 'Pas', 'Rec', 'Pos', 'Neg', 'Ges']
    confusion_matrix = MatrixMeter(label_names)
    global top_acc

    pred_result = dict()

    with torch.no_grad():
        for data in tqdm(dataloader):
            if args.model == 'di3d':
                input_f, input_s, labels, _ = data
                input_f, input_s = input_f.cuda(), input_s.cuda()
                labels = labels.cuda(non_blocking=True)
                output = model(input_f, input_s)

            else:
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
            target = dataloader.dataset.id2label[i.item()]
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

        if confusion_matrix.acc > top_acc and args.main_rank:
            top_acc = confusion_matrix.acc
            save(model, 'best')

        save(model, 'last')

        if logger is not None:
            logger.add_scalar('val/top1', 100*confusion_matrix.acc, epoch)
            logger.add_scalar('val/top2', 100*top2.avg, epoch)
            logger.add_figure('val/confusion',
                              draw_confusion_matrix(confusion_matrix._data, label_names), epoch, close=False)

        print("Top1:%.2f Top2:%.2f" % (100*confusion_matrix.acc, 100*top2.avg))


def save(model, comment):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'state_dict': state_dict, 'args': args, 'top_acc': top_acc},
               '%s_split_%d_%s_%s_%s_%s.pt' % ('pev', split_idx, args.model, args.mode, comment, args.view))


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


if __name__ == '__main__':
    # need to add argparse
    if args.eval:
        evaluate(batch_size=args.batch_size)
    else:
        run(max_steps=args.epochs, mode=args.mode,
            batch_size=args.batch_size, save_model=args.save_model)
