import torch
import inceptionv3
import torch.nn as nn
from BilinearPooling import LRBilinearPooling
import torch.nn.functional as F

class BPTSN(nn.Module):
    def __init__(self, num_classes, in_channels=3, **kwargs):
        super(BPTSN, self).__init__()
        self.backbone = inceptionv3.inception_v3(pretrained=True, aux_logits=False,
                                                 feature_extract=True, **kwargs)
        if in_channels == 2:
            self.backbone.Conv2d_1a_3x3 = inceptionv3.BasicConv2d(
                2, 32, kernel_size=3, stride=2)
        self.lrbp = LRBilinearPooling(768, 512, num_classes)

    def forward(self, x):
        """forward 

        Args:
            x (Tensor): Shape N C T H W
        """
        x = x.permute(0, 2, 1, 3, 4)
        shape = list(x.shape)
        x = x.reshape([-1]+shape[2:])
        # x Shape N*T CHW
        x = self.backbone(x)
        x = x.reshape(shape[0], shape[1], 768, 17, 17)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(shape[0], 768, -1)
        x = F.dropout(x)
        x = self.lrbp(x)
        # x Shape N*T 768 17 17
        return x
