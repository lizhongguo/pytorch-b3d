import torch
import torch.nn as nn
from BilinearPooling import SRBilinearPooling, BilinearPooling
import torch.nn.functional as F
class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, in_channels=3, pretrained=False):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.Conv3d(64, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2a = nn.Conv3d(
            64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.conv2b = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.Conv3d(
            128, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3a = nn.Conv3d(
            128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.Conv3d(
            256, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(256)


        #self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.pool4 = nn.Conv3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        #self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        #self.pool5 = nn.Conv3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        #self.fc6 = nn.Linear(8192, 4096)
        #self.fc7 = nn.Linear(4096, 4096)
        #self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2a(x)))
        #x = self.relu(self.conv2b(x))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3a(x)))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        return x
        '''
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits
        '''

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class BPC3D(nn.Module):
    def __init__(self, num_classes, in_channels=3, **kwargs):
        super(BPC3D, self).__init__()
        self.backbone = C3D(num_classes, in_channels=in_channels, **kwargs)
        #self.lrbp = LRBilinearPooling(256+320+128+128, 512, num_classes)
        self.conv = nn.Conv3d(256, 64, kernel_size=1)
        self.bp = BilinearPooling()
        self.fc = nn.Linear(64*64, num_classes)
        self.bn_1 = nn.BatchNorm3d(64)
        self.bn_2 = nn.BatchNorm1d(64*64)

    def forward(self, x):
        """forward 

        Args:
            x (Tensor): Shape N C T H W
        """
        x = self.backbone(x)
        x = self.conv(x)
        x = self.bn_1(x)
        x = self.bp(x)
        x = self.bn_2(x)
        x = F.dropout(x, p=0.8)
        x = self.fc(x)
        return x
