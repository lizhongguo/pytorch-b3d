import torch
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch.nn.functional as F

class DI3D(nn.Module):
    def __init__(self, num_classes, in_channels=3, mode='cat', **kwargs):
        super(DI3D, self).__init__()
        self.backbone_f = InceptionI3d(num_classes, in_channels=in_channels, extract_feature=True, **kwargs)
        self.backbone_s = InceptionI3d(num_classes, in_channels=in_channels, extract_feature=True,**kwargs)
        
        self.weight = self.backbone_f.logits.conv3d.weight.squeeze().detach().cuda()
        self.weight = F.softmax(self.weight, dim=1)
        self.weight.requires_grad = False
        #self.weight.data.uniform_()
        
        self.fc = nn.Linear(1024, num_classes)
    
        #self.mode = mode
    '''
    def forward(self, x1, x2):
        """forward 

        Args:
            x (Tensor): Shape N C T H W
        """
        feature_f = self.backbone_f(x1)
        feature_s = self.backbone_s(x2)
        #print(feature_f.shape)
        #print(feature_s.shape)
        if len(feature_f.shape) < 2:
            feature_f = feature_f.reshape([1,]+list(feature_f.shape))
            feature_s = feature_s.reshape([1,]+list(feature_s.shape))

        #w = torch.sigmoid(self.weight)
        #print(feature_f.shape, feature_s.shape, w.shape)
        #feature = torch.einsum('nc,c->nc', feature_f, w) + torch.einsum('nc,c->nc', feature_s, 1. - w)
        if self.mode == 'cat':
            feature = torch.cat((feature_f,feature_s),dim=1)            
        feature = self.fc(feature)
        return feature
    '''
    def mse(self, f, s, labels):
        w = self.weight.index_select(dim=0, index=labels)
        err = f - s
        mse_err = 8.*torch.einsum('nc,nc->nc', w, torch.einsum('nc, nc->nc', err, err)).sum()
        return mse_err

    def grad_guided_mse(self, f, s, labels):
        err = f - s
        mse_err = 1.*torch.sqrt(torch.einsum('nc, nc->nc', err, err).mean())
        return mse_err


    def forward(self, x1, x2, labels=None):
        """forward 

        Args:
            x (Tensor): Shape N C T H W
        """
        if self.training:
            with torch.no_grad():
                feature_f = self.backbone_f(x1)
            feature_s = self.backbone_s(x2)
            #print(feature_f.shape)
            #print(feature_s.shape)
            if len(feature_f.shape) < 2:
                feature_f = feature_f.reshape([1,]+list(feature_f.shape))
                feature_s = feature_s.reshape([1,]+list(feature_s.shape))

            loss_mse = self.mse(feature_f, feature_s, labels)

            feature = self.fc(feature_s)
            return feature, loss_mse

        else:
            feature_s = self.backbone_s(x2)
            #print(feature_f.shape)
            #print(feature_s.shape)
            if len(feature_s.shape) < 2:
                feature_s = feature_s.reshape([1,]+list(feature_s.shape))

            feature = self.fc(feature_s)
            return feature
