import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split',  type=int, default=1)
parser.add_argument('--mode',  type=str, default='rgb')
parser.add_argument('--model',  type=str, default='i3d')
args = parser.parse_args()

for subset in ['train', 'val']:
    split_list = open('/home/lizhongguo/dataset/pev_split/%s_split_%d.txt'
                      % (subset, args.split))

    id2label = dict()
    for s in split_list:
        s = s.split(' ')
        label = int(s[2])
        id2label[int(s[0][:-2])] = label

    first_view = {}
    data = torch.load('pev_split_%d_%s_%s_%s_%s_scores.pt' %
                      (args.split, args.model, args.mode, 'fs', subset))
    split_txt = open('pev_split_%d_%s_%s_%s_svm.txt' %
                     (args.split, args.model, args.mode, subset), 'w')
    for l in data:
        label = id2label[l[1].item()]
        feature = [e for e in l[0]]
        formattxt = ('%d %s\n') % (label, ' '.join(
            ['%d:%.4f' % (i+1, e) for i, e in enumerate(feature)]))
        split_txt.write(formattxt)
