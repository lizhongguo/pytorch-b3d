import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
import pdb
import random

def pil_loader(path, mode):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if mode == 'rgb':
                return img.convert('RGB')
            else:
                return img.convert('L')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, mode='rgb', image_loader=None):
    video = []
    for i in frame_indices:

        if mode == 'rgb':
            '''
            npy_path = os.path.join(video_dir_path, '{:05d}.npy'.format(i))
            if os.path.exists(npy_path):
                video.append(Image.fromarray(np.load(npy_path)))
                continue
            '''
            image_path = os.path.join(
                video_dir_path, 'img_{:05d}.jpg'.format(i))
            video.append(image_loader(image_path, mode))

        elif mode == 'flow_x':
            image_path = os.path.join(
                video_dir_path, 'flow_x_{:05d}.jpg'.format(i))
            video.append(image_loader(image_path, mode))

        elif mode == 'flow_y':
            image_path = os.path.join(
                video_dir_path, 'flow_y_{:05d}.jpg'.format(i))
            video.append(image_loader(image_path, mode))

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, sample_freq=1):

    # annotation[video_idx] = label
    split_list = open(annotation_path)

    dataset = []
    id2label = dict()
    for i, s in enumerate(split_list):

        if i % 300 == 0:
            print('dataset loading [{}/{}]'.format(i, 'None'))

        s = s.split(' ')
        video_name = s[0]
        label = int(s[2])

        # only use second person's view
        video_path = os.path.join(root_path, video_name)
        if not os.path.exists(video_path):
            continue
        # all video's length is 90 frames
        n_frames = 89
        n_frames = n_frames // sample_freq

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': i
        }
        sample['label'] = label

        id2label[i] = label

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.floor((n_frames - 1 - sample_duration) /
                                      (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, max(2, n_frames - sample_duration + 1), step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, id2label

# use undersample method to avoid class imbalance


def make_raw_dataset(annotation_path, subset):

    # annotation[video_idx] = label
    split_list = open(annotation_path)

    raw_dataset = dict()
    min_class_len = 0

    for i, s in enumerate(split_list):
        s = s.split(' ')
        video_name = s[0]
        label = int(s[2])

        if label not in raw_dataset:
            raw_dataset[label] = list()

        raw_dataset[label].append(video_name)
        min_class_len = min_class_len + 1

    for e in raw_dataset:
        if len(raw_dataset[e]) < min_class_len:
            min_class_len = len(raw_dataset[e])
    print("min class len %d" % min_class_len)
    return raw_dataset, min_class_len


class PEV(data.Dataset):
    """
    Args:
    root (string): Root directory path.
    spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
    target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 sample_freq=1,
                 mode='rgb'):
        # self.data = make_dataset(
        #    root_path, annotation_path, subset, n_samples_for_each_video,
        #    sample_duration)

        self.root_path = root_path
        self.annotation_path = annotation_path
        self.n_samples_for_each_video = n_samples_for_each_video
        self.sample_duration = sample_duration
        self.subset = subset
        self.sample_freq = sample_freq
        self.mode = mode
        if subset in ['training', 'validation']:
            self.raw_data, self.min_class_len = make_raw_dataset(
                annotation_path, subset)

            self.undersample(self.root_path, self.raw_data, self.subset,
                             self.min_class_len, self.n_samples_for_each_video, self.sample_duration, sample_freq)

            self.labels = self.raw_data.keys()

        else:
            self.data, self.id2label = make_dataset(
                root_path, annotation_path, subset, n_samples_for_each_video,
                sample_duration, sample_freq)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.random_select = False
        '''
        self.target_matrix = [[0, 1, 0, 0, 1, 0, 0],
                               [1, 0, 0, 0, 0, 1, 1],
                               [2, 0, 1, 1, 1, 1, 0],
                               [3, 0, 2, 1, 1, 1, 0],
                               [4, 0, 0, 0, 0, 0, 2],
                               [5, 0, 0, 0, 0, 0, 3],
                               [6, 1, 0, 0, 0, 0, 0]]
        self.multi_label_shape = [7, 2, 3, 2, 2, 2, 4]
        '''
        # self.target_matrix = [[0, 0, 0],
        #                      [1, 0, 0],
        #                      [2, 0, 0],
        #                      [3, 0, 0],
        #                      [4, 1, 0],
        #                      [5, 0, 1],
        #                      [6, 2, 2]]
        # self.multi_label_shape = [7, 3, 3]

        # self.multi_label_num = 1
        # for e in self.multi_label_shape:
        #    self.multi_label_num = self.multi_label_num * e
        # self.multi_label_data = None
        # self._multilabel_initialize()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        if self.sample_freq > 1:
            if self.random_select:
                frame_indices = [ random.randint(idx*self.sample_freq, min((idx+1)*self.sample_freq, 89)) for idx in frame_indices]
            else:
                frame_indices = [idx*self.sample_freq for idx in frame_indices]


        if self.mode == 'rgb':
            clip = self.loader(path, frame_indices, 'rgb')
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        elif self.mode == 'flow':

            flow_x = self.loader(path, frame_indices, 'flow_x')
            flow_y = self.loader(path, frame_indices, 'flow_y')

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                flow_x = [self.spatial_transform(img) for img in flow_x]
                flow_y = [self.spatial_transform(img) for img in flow_y]

            flow_x = torch.stack(flow_x, 0)
            flow_y = torch.stack(flow_y, 0)
            clip = torch.cat((flow_x, flow_y), dim=1)
            clip = clip.permute(1, 0, 2, 3)

        elif self.mode == 'rgb+flow':
            rgb = self.loader(path, frame_indices, 'rgb')
            flow_x = self.loader(path, frame_indices, 'flow_x')
            flow_y = self.loader(path, frame_indices, 'flow_y')

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                rgb = [self.spatial_transform(img) for img in rgb]
                flow_x = [self.spatial_transform(img) for img in flow_x]
                flow_y = [self.spatial_transform(img) for img in flow_y]

            flow_x = torch.stack(flow_x, 0)
            flow_y = torch.stack(flow_y, 0)
            rgb = torch.stack(rgb, 0)
            clip = torch.cat((rgb, flow_x, flow_y), dim=1)
            clip = clip.permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target, index

    def __len__(self):
        return len(self.data)

    def _multilabel_initialize(self):
        if self.multi_label_data is None:
            for e in self.data:
                e['label'] = self.target_matrix[e['label']]

    def update_label(self, data_index, dim_index, prob):

        for data_idx, p in zip(data_index, prob):
            # p means p(Action=gt, TargetLabel)
            if isinstance(self.data[data_idx]['label'], int):
                self.data[data_idx]['label'] = [
                    self.data[data_idx]['label']]*len(self.multi_label_shape)

            p = p[self.data[data_idx]['label'][0]]
            p = (p/p.sum()).cpu().numpy()
            # now p is p(Target|Action=gt)
            l = np.random.choice(np.arange(len(p)), p=p)
            self.data[data_idx]['label'][dim_index] = l.item()

    def undersample(self, root_path, raw_dataset, subset, min_class_len, n_samples_for_each_video,
                    sample_duration, sample_freq=1):
        dataset = []

        v_id = 0
        for i, label in enumerate(raw_dataset):

            videos = raw_dataset[label]

            for video_name in np.random.choice(videos, min_class_len, replace=False):
                # only use second person's view
                video_path = os.path.join(root_path, video_name)
                if not os.path.exists(video_path):
                    continue
                # all video's length is 90 frames
                n_frames = 89
                n_frames = n_frames // sample_freq

                begin_t = 1
                end_t = n_frames
                sample = {
                    'video': video_path,
                    'segment': [begin_t, end_t],
                    'n_frames': n_frames,
                    'video_id': v_id
                }
                sample['label'] = label

                v_id = v_id + 1

                if n_samples_for_each_video == 1:
                    sample['frame_indices'] = list(range(1, n_frames + 1))
                    dataset.append(sample)
                else:
                    if n_samples_for_each_video > 1:
                        step = max(1,
                                   math.floor((n_frames - 1 - sample_duration) /
                                              (n_samples_for_each_video - 1)))
                    else:
                        step = sample_duration
                    for j in range(1, max(2, n_frames - sample_duration + 1), step):
                        sample_j = copy.deepcopy(sample)
                        sample_j['frame_indices'] = list(
                            range(j, min(n_frames + 1, j + sample_duration)))
                        dataset.append(sample_j)

        self.data = dataset
