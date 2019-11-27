import cv2
import numpy as np

import torch
import torch.nn as nn

import random
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
from DI3D import DI3D, MBI3D

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
parser.add_argument('--fuse', type=str, default='cat',
                    choices=['cat', 'cbp', 'cbp+cat'])
parser.add_argument('--eval', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--model', type=str,
                    choices=['i3d', 'r2plus1d', 'w3d', 'tsn', 'bptsn', 'bpi3d', 'bpc3d', 'di3d', 'mbi3d'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sample_freq', type=int, default=1)
parser.add_argument('--sample_step', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=6,
                    help='num of samples for each video')
parser.add_argument('--clip_len', type=int, default=32)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--print_loss', action='store_true')
parser.add_argument('--view', type=str, choices=['s', 'f', 'fs'])
parser.add_argument('--split_idx', type=int, choices=[1, 2, 3])


args = parser.parse_args()

split_idx = args.split_idx

data_root = '/home/lizhongguo/dataset/pev_of'
torch.manual_seed(123)
np.random.seed(123)

# integrated gradients


def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline=None, steps=50, cuda=True):
    if baseline is None:
        baseline = [0 * i for i in inputs]
    # scale inputs and compute gradients
    scaled_inputs = [[bl + (float(i) / steps) * (sinput - bl)
                      for i in range(0, steps + 1)] for sinput, bl in zip(inputs, baseline)]
    grads, _ = predict_and_gradients(
        scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    print(avg_grads.shape)
    integrated_grad = [(i.cpu().numpy() - b.cpu().numpy())
                       * ag for i, b, ag in zip(inputs, baseline, avg_grads)]
    return integrated_grad


def calculate_outputs_and_gradients(inputs, model, target_label_idx, cuda=False):
    # do the pre-processing
    gradients = []
    for input_f, input_s in zip(inputs[0], inputs[1]):
        input_f.requires_grad = True
        input_s.requires_grad = True
        output = model(input_f, input_s)
        loss = F.cross_entropy(output, target_label_idx)
        # clear grad
        model.zero_grad()
        loss.backward()
        # print(input_f.grad)
        gradient = [input_f.grad.detach().cpu().numpy(),
                    input_s.grad.detach().cpu().numpy()]
        gradients.append(gradient)
        model.zero_grad()

    return np.array(gradients), target_label_idx


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2, plot_distribution=False):
    m = compute_threshold_by_top_percentage(
        attributions, percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(
        attributions, percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (transformed >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError
    return threshold


def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 1)


def visualize(attributions, image, positive_channel=[255, 255, 0], negative_channel=[255, 0, 0], polarity='positive',
              clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False,
              structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True,
              mask_mode=False, plot_distribution=False):
    if polarity == 'both':
        raise NotImplementedError

    elif polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
        channel = positive_channel

    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(attributions, clip_above_percentile,
                                    clip_below_percentile, 0.0, plot_distribution=plot_distribution)
    attributions_mask = attributions.copy()
    if morphological_cleanup:
        raise NotImplementedError
    if outlines:
        raise NotImplementedError
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 2)
            attributions = np.clip(attributions * image, 0, 1)
    return attributions




def evaluate(max_steps=320):

    if args.model == 'i3d' or args.model == 'bpi3d' or args.model == 'di3d' or args.model == 'mbi3d':
        scale_size = 224
    elif args.model == 'r2plus1d' or args.model == 'w3d' or args.model == 'bpc3d':
        scale_size = 112
    elif args.model == 'tsn' or args.model == 'bptsn':
        scale_size = 224  # 299
    else:
        raise Exception('Model %s not implemented' % args.model)

    if args.model == 'i3d' or args.model == 'r2plus1d' or args.model == 'bpi3d' or args.model == 'bpc3d' \
            or args.model == 'di3d' or args.model == 'mbi3d':
        if args.mode == 'rgb':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        elif args.mode == 'flow':
            mean = [0.5, 0.5]
            std = [0.03, 0.03]
        elif args.mode == 'rgb+flow':
            mean = [0.5, 0.5, 0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5, 0.03, 0.03]

    elif args.model == 'tsn' or args.model == 'bptsn':
        if args.mode == 'rgb':
            mean = [104./255., 117./255., 128./255.]
            std = [1., 1., 1.]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [1., 1., 1.]

    else:
        raise Exception('Model %s not implemented' % args.model)

    train_transforms = Compose([MultiScaleRandomCrop([1.0, 0.95, 0.95*0.95], scale_size),
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

    val_temporal_transforms = Compose([TemporalBeginCrop(clip_len),
                                       RepeatPadding(clip_len)])
    val_dataset = PEV(
        data_root,
        '/home/lizhongguo/dataset/pev_split/val_split_%d.txt' % split_idx,
        'evaluation',
        args.n_samples,
        spatial_transform=test_transforms,
        temporal_transform=val_temporal_transforms,
        target_transform=ClassLabel(),
        sample_duration=clip_len, sample_freq=args.sample_freq, mode=args.mode, sample_step=args.sample_step, view=args.view)

    val_sampler = None

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None), num_workers=2, pin_memory=True, sampler=val_sampler, drop_last=False)

    # setup the model
    if args.model == 'mbi3d':
        model = MBI3D(7, args.fuse, ['f%s' % args.mode, 's%s' % args.mode])
        checkpoint = torch.load(
            'pev_split_%d_%s_%s_best%sfs.pt' % (args.split_idx, args.model, args.mode,
                                                '_' if args.fuse == 'cbp' else '_'+args.fuse+'_'), map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        model.train(False)
        model.eval()

    for data in val_dataloader:
        input_f, input_s, labels, index = data
        input_f, input_s = input_f.cuda(), input_s.cuda()
        labels = labels.cuda(non_blocking=True)
        #baseline_f = torch.zeros_like(input_f)
        #baseline_s = torch.zeros_like(input_s)
        #output = model(input_f, input_s)
        ig = integrated_gradients(
            (input_f, input_s), model, labels, calculate_outputs_and_gradients)
        ig = [np.transpose(ig[0],(0, 2, 3, 4, 1)), np.transpose(ig[1],(0, 2, 3, 4, 1))]

        m = 0.5
        v = 0.03 if args.mode == 'flow' else 0.5
        input_f.div_(1/v).sub_(-m)
        input_s.div_(1/v).sub_(-m)
        input_f = input_f.permute(0, 2, 3, 4, 1)
        input_s = input_s.permute(0, 2, 3, 4, 1)

        video_id = val_dataset.data[index]['video_id']

        for attr_f, attr_s, input_f, input_s in zip(ig[0], ig[1], input_f, input_s):
            # N
            idx = 0
            for a_f, a_s, i_f, i_s in zip(attr_f, attr_s, input_f, input_s):
                idx += 1
                img_integrated_gradient_overlay = visualize(a_f, i_f.cpu().numpy(), clip_above_percentile=99, clip_below_percentile=90,
                                                            overlay=True, mask_mode=False)
                cv2.imwrite('vis/ig_%05d_f_%d.jpg' %
                            (video_id, idx), 255*img_integrated_gradient_overlay[:,:,(2,1,0)])
                img_integrated_gradient_overlay = visualize(a_s, i_s.cpu().numpy(), clip_above_percentile=99, clip_below_percentile=90,
                                                            overlay=True, mask_mode=False)
                cv2.imwrite('vis/ig_%05d_s_%d.jpg' %
                            (video_id ,idx), 255*img_integrated_gradient_overlay[:,:,(2,1,0)])


if __name__ == '__main__':
    evaluate()
