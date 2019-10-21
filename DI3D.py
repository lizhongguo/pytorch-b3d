import torch
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch.nn.functional as F
from CompactBilinearPooling import CountSketch, CompactBilinearPooling

class DI3D(nn.Module):
    def __init__(self, num_classes, in_channels=3, mode='cat', modal='f+s', **kwargs):
        super(DI3D, self).__init__()

        self.backbone_f = InceptionI3d(
            num_classes, in_channels=in_channels, extract_feature=True, **kwargs)
        self.backbone_s = InceptionI3d(
            num_classes, in_channels=in_channels, extract_feature=True, **kwargs)
        self.dropout = nn.Dropout(p=0.5)

        if mode == 'cbp':
            self.mcb = CompactBilinearPooling(1024, 1024, 1024) 
            self.fc = nn.Linear(1024, num_classes)

        elif mode == 'cat':
            self.fc = nn.Linear(2048, num_classes)

        elif mode == 'cbp+cat':
            self.mcb = CompactBilinearPooling(1024, 1024, 1024) 
            self.fc = nn.Linear(3072, num_classes)

        self.mode = mode

    def forward(self, x_f, x_s, labels=None):
        feature_f = self.backbone_f(x_f)
        feature_s = self.backbone_s(x_s)
        
        if self.mode == 'cbp':
            #feature_f = torch.sigmoid(feature_f)
            #feature_s = torch.sigmoid(feature_s)
            #feature_f = self.mcb(feature_f, feature_f)
            #feature_s = self.mcb(feature_s, feature_s)
            feature = self.mcb(feature_f, feature_s)
            feature = self.dropout(feature)

        elif self.mode == 'cat':
            feature = torch.cat((feature_f, feature_s), dim=1)
            feature = self.dropout(feature)

        elif self.mode == 'cbp+cat':
            feature_cbp = self.mcb(feature_f, feature_s)
            feature = torch.cat((feature_f, feature_s, feature_cbp), dim=1)
            feature = self.dropout(feature)
    
        y = self.fc(feature)
        return y
