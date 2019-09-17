import torch
import torch.nn as nn
from DWT import DWT3D
from torch.nn.modules.utils import _triple
import math
from torchvision.utils import make_grid

class WaveletEncoding(nn.Module):
    def __init__(self, dim='thw'):
        super(WaveletEncoding, self).__init__()
        self.dwt = DWT3D(wave='db1', dim=dim)

    def forward(self, x):
        x = self.dwt(x)
        return x


class ConvPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPooling, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              (3, 3, 3), (2, 2, 2), (1, 1, 1))

    def forward(self, input):
        y = self.conv(input)
        return y


class WaveletPooling(nn.Module):
    def __init__(self, in_channels, out_channels, dim='thw'):
        super(WaveletPooling, self).__init__()
        self.conv = nn.Conv3d(in_channels*(2**len(dim)),
                              out_channels, (1, 1, 1))
        self.dwt = DWT3D(wave='db4', dim=dim)

    def forward(self, input):
        y = self.conv(self.dwt(input))
        return y


class W3D(nn.Module):
    """
    The W3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(W3D, self).__init__()

        # 32 112 112
        self.conv1a = nn.Conv3d(
            3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv1b = nn.Conv3d(
            64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        self.pool1 = WaveletEncoding(dim='t')

        # 16 112 112
        self.conv2a = nn.Conv3d(
            128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2b = nn.Conv3d(
            256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # 8 56 56
        self.conv3a = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(
            512, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.pool3 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # 4 28 28

        self.conv4a = nn.Conv3d(
            1024, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(
            1024, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # 2 14 14
        self.conv5a = nn.Conv3d(
            1024, 1024, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 14, 14))

        self.logits = nn.Conv3d(
            1024, num_classes, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1))

        self.dropout = nn.Dropout(p=0.5)
        self.logger = None

    def forward(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        if self.logger is not None:
            y = x[:, 0:8, 1, :, :].detach().reshape(-1,1,112,112)
            self.logger.add_image('training/feature_1',
                                  make_grid(y, normalize=True))
            y = x[:, 64:72, 1, :, :].detach().reshape(-1,1,112,112)
            self.logger.add_image('training/feature_2',
                                  make_grid(y, normalize=True))

            # self.logger.add_video('training/feature_1',y)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)

        x = self.pool4(x)

        x = self.conv5a(x)
        #x = self.conv5b(x)
        #x = self.bn5(x)
        x = self.pool5(x)

        x = self.dropout(x)
        logits = self.logits(x)
        logits = logits.squeeze()

        return logits


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 32, 224, 224).cuda()
    inputs.requires_grad = True
    net = W3D(7).cuda()
    outputs = net(inputs)
    l = outputs.sum()
    print(l)
    l.backward()
