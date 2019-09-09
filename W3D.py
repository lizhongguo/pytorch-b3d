import torch
import torch.nn as nn
from DWT import DWT3D


class WaveletEncoding(nn.Module):
    def __init__(self, dim='thw'):
        super(WaveletEncoding, self).__init__()
        self.dwt = DWT3D(wave='haar', dim=dim)

    def forward(self, x):
        x = self.dwt(x)
        return x


class W3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(W3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = WaveletEncoding()

        self.bn1 = nn.BatchNorm3d(4, eps=0.001, momentum=0.01)

        self.conv2 = nn.Conv3d(
            32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = WaveletEncoding()
        self.bn2 = nn.BatchNorm3d(32, eps=0.001, momentum=0.01)

        self.conv3a = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = WaveletEncoding()
        self.bn3 = nn.BatchNorm3d(256, eps=0.001, momentum=0.01)

        self.conv4a = nn.Conv3d(
            2048, 2048, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(
            2048, 2048, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2) )
        self.bn4 = nn.BatchNorm3d(2048, eps=0.001, momentum=0.01)

        self.conv5a = nn.Conv3d(
            2048, 4096, kernel_size=(4, 1, 1) )

        self.bn5 = nn.BatchNorm3d(4096, eps=0.001, momentum=0.01)
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 7, 7))

        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.bn4(x)

        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.bn5(x)

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
