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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
parser.add_argument('--save_model', type=str)
parser.add_argument('--root', type=str)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--model', type=str, choices=['i3d', 'r2plus1d', 'w3d'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sample_freq', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=6,
                    help='num of samples for each video')
parser.add_argument('--clip_len', type=int, default=32)
parser.add_argument('--resume', type=str, default=None)


args = parser.parse_args()
top_acc = 0

from torch.nn.parallel.distributed import DistributedDataParallel as DDP

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# args.gpu = 0
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

# print('rank %d started' % torch.distributed.get_rank())

args.main_rank = (args.distributed and torch.distributed.get_rank()
                  == 0) or (not args.distributed)
data_root = '/home/lizhongguo/dataset/pev_of'


def model_builder():
    # setup the model
    if args.model == 'i3d':
        if args.mode == 'flow':
            model = InceptionI3d(num_classes=7, in_channels=2)
            model.load_state_dict({k: v for k, v in torch.load('models/flow_imagenet.pt').items()
                                   if k.find('logits') < 0}, strict=False)
        else:
            model = InceptionI3d(num_classes=7,
                                 in_channels=3, dropout_keep_prob=0.5)
            model.load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                   if k.find('logits') < 0}, strict=False)
    elif args.model == 'r2plus1d':
        model = R2Plus1DClassifier(num_classes=7)
    elif args.model == 'w3d':
        model = W3D(num_classes=7)
        # model.load_state_dict(torch.load('pev_i3d_best.pt'))

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

    if args.distributed:
        lr = args.lr * args.batch_size * args.world_size / 64.
    else:
        lr = args.lr * args.batch_size / 56.

    # lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60])
    if args.distributed:
        model = DDP(model)
    else:
        model = nn.DataParallel(model)

    return model

class L21(nn.Module):
    def __init__(self):
        super(L21, self).__init__()

    def forward(self, x):
        x = torch.einsum('ncthw,ncthw->nt',x,x)
        x = torch.einsum('nt->n', torch.sqrt(x))
        return x

def attack(init_lr=0.1, max_steps=320, mode='rgb', batch_size=20, save_model=''):

    logger = SummaryWriter()

    # setup dataset
    if args.model in ('i3d', ):
        scale_size = 224
    elif args.model in ('r2plus1d', 'w3d'):
        scale_size = 112
    else:
        raise Exception('Model %s not implemented' % args.model)

    test_transforms = Compose([MultiScaleRandomCrop([1.0], scale_size),
                               ToTensor(255.0),
                               Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])
    clip_len = args.clip_len

    temporal_transforms = Compose(
        [TemporalCenterCrop(clip_len), RepeatPadding(clip_len)])
    target_transforms = ClassLabel()
    val_dataset = PEV(
        data_root,
        '/home/lizhongguo/dataset/pev_split/val_split_3.txt',
        'evaluation',
        args.n_samples,
        spatial_transform=test_transforms,
        temporal_transform=temporal_transforms,
        target_transform=target_transforms,
        sample_duration=clip_len, sample_freq=args.sample_freq, mode=args.mode)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # setup the model
    model = model_builder()
    model.requires_grad_(False)


    criterion = nn.CrossEntropyLoss().cuda()
    l21 = L21()

    step = 0
    #for data in tqdm(val_dataloader):
    data = val_dataset.__getitem__(13)
    inputs, labels, _ = data
    labels = torch.from_numpy(labels).view([-1])
    inputs = inputs.view([1,] + list(inputs.size()))
    inputs = inputs.cuda()
    labels = labels.cuda(non_blocking=True)

    pertubation = 1e-4*torch.ones_like(inputs)

    pertubation.requires_grad=True
    optimizer = optim.SGD([pertubation], lr=args.lr)

    mask = torch.zeros_like(inputs)
    mask[:,:,:,:,:] = 1e-4
    mask[:,:,:,:16,:16] = 1.

    n_iter = 300
    
    for step in range(n_iter):

        optimizer.zero_grad()

        pertubation_masked = pertubation.mul(mask)

        inputs_perb = torch.clamp(inputs + pertubation_masked,-1., 1.)
        output = model(inputs_perb)

        cls_loss = criterion(output, labels)
        l21_loss = l21(pertubation_masked)
        loss = - cls_loss + l21_loss

        pertubation_vis = pertubation_masked.detach().permute(0,2,1,3,4).reshape(-1, 1, 224, 224)
        logger.add_image('attack/perb', torchvision.utils.make_grid(pertubation_vis), step)
        logger.add_scalar('attack/loss', loss.item(), step)
        logger.add_scalar('attack/cls_loss', cls_loss.item(), step)

        print('Iteration %d Loss %f' % (step, loss.item()))
        print('Iteration %d ClsLoss %f' % (step, cls_loss.item()))
        print('Iteration %d L21Loss %f' % (step, l21_loss.item()))

        loss.backward()

        #print(pertubation.grad)
        optimizer.step()

        #print(pertubation)

    logger.close()



if __name__ == '__main__':
    # need to add argparse
    attack(batch_size=args.batch_size)
