import torch
import torch.nn as nn
import torch.nn.functional as F


class LRBilinearPooling(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(LRBilinearPooling, self).__init__()

        self.U_p = nn.Parameter(torch.randn(
            out_channels, mid_channels, in_channels))
        self.U_n = nn.Parameter(torch.randn(
            out_channels, mid_channels, in_channels))
        nn.init.xavier_normal_(self.U_p)
        nn.init.xavier_normal_(self.U_n)

    def F_square_norm(self, x):
        return torch.einsum('bomn,bomn->bo', x, x)

    def mul(self, x, y):
        return torch.einsum('omi,bin->bomn', x, y)

    def forward(self, x):
        shape = list(x.shape)
        x = x.reshape(shape[0], shape[1], -1)
        return self.F_square_norm(self.mul(self.U_p, x)) - \
            self.F_square_norm(self.mul(self.U_n, x))


class BilinearPooling(nn.Module):
    def __init__(self):
        super(BilinearPooling, self).__init__()
    def forward(self, x):
        shape = list(x.shape)
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.einsum('bxn,byn->bxy', x, x)
        x = x.reshape(shape[0], -1)
        x = torch.sign(x) * torch.sqrt(x.abs())
        return x

class SRBilinearPooling(nn.Module):
    def __init__(self, spatial_dims):
        """SRBilinearPooling Spatial Relative Bilinear Pooling
        Args:
            in_channels (int): 
            mid_channels (int): 
            spatial_dims (int): dims of space
        """
        super(SRBilinearPooling, self).__init__()
        self.sr = nn.Parameter(torch.randn(
            spatial_dims, spatial_dims))
        nn.init.xavier_uniform_(self.sr)


    def forward(self, x):
        shape = list(x.shape)
        x = x.reshape(shape[0], shape[1], -1)
        y = torch.einsum('bxn, nm->bxm', x, self.sr)
        out = torch.einsum('bxm, bym->bxy', y, x)
        out = out.reshape(shape[0],-1)
        return out



class LRBilinearLoss(nn.Module):
    def __init__(self, n_classes=7, type='MM'):
        super(LRBilinearLoss, self).__init__()
        self.n_classes = n_classes
        self.type = type
        #self.b = nn.Parameter(torch.FloatTensor(0))

    def forward(self, input, target):
        """forward Local Rank Bilinear Loss
            Original Paper used Hinge Loss For single-label single-classification SVM
            Choose CrossEntropy For single-lable multi-classification
        Args:
            output (Tensor): shape [N, classes]
            target (Tensor): shape [N]
        """
        if self.type == 'CE':
            return F.cross_entropy(input, target)
        else:
            return F.multi_margin_loss(input, target)
