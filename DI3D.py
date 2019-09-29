import torch
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch.nn.functional as F
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

class DI3D(nn.Module):
    def __init__(self, num_classes, in_channels=3, mode='cat', **kwargs):
        super(DI3D, self).__init__()

        self.backbone_rgb = InceptionI3d(num_classes, in_channels=3, extract_feature=True, **kwargs)
        self.backbone_flow = InceptionI3d(num_classes, in_channels=3, extract_feature=True,**kwargs)
        self.mcb = CompactBilinearPooling(1024, 1024, 1024)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x_rgb, x_flow, labels=None):
        feature_rgb = self.backbone_rgb(x_rgb)
        feature_flow = self.backbone_flow(x_flow)
        feature = self.mcb(feature_rgb, feature_flow)
        y = self.fc(feature)
        return y