import torch
import torch.nn as nn
from DWT import DWT3D
from torch.nn.modules.utils import _triple
import math


class WaveletEncoding(nn.Module):
    def __init__(self, dim='thw'):
        super(WaveletEncoding, self).__init__()
        self.dwt = DWT3D(wave='db4', dim=dim)

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


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) /
                                           (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.sbn = nn.BatchNorm3d(intermed_channels)

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)
        self.tbn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.sbn(self.spatial_conv(x))
        x = self.tbn(self.temporal_conv(x))
        return x


class W3D(nn.Module):
    """
    The W3D network.
    """

    def __init__(self, num_classes, pretrained=False, Conv3d=SpatioTemporalConv):
        super(W3D, self).__init__()

        # 32 112 112
        self.conv1 = Conv3d(3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool1 = WaveletEncoding(dim='t')

        # 16 112 112
        self.conv2a = Conv3d(
            128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2b = Conv3d(
            128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = WaveletEncoding()

        # 8 56 56

        self.conv3a = Conv3d(
            1024, 1024, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv3b = Conv3d(
            1024, 1024, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv3c = Conv3d(
            1024, 2048, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # 4 28 28

        self.conv4a = Conv3d(
            2048, 2048, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv4b = Conv3d(
            2048, 2048, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # 2 14 14
        self.conv5a = Conv3d(
            2048, 4096, kernel_size=(2, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 14, 14))

        self.fc8 = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        #x = self.conv5a(x)
        #x = self.conv5b(x)
        #x = self.bn5(x)
        x = self.pool5(x)

        x = x.squeeze()
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits


if __name__ == "__main__":
    inputs = torch.rand(8, 3, 32, 224, 224).cuda()
    inputs.requires_grad = True
    net = W3D(7).cuda()
    outputs = net(inputs)
    l = outputs.sum()
    print(l)
    l.backward()
