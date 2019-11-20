import torch
from pytorch_i3d import InceptionI3d
import torch.nn as nn
import torch.nn.functional as F
#from CompactBilinearPooling import CompactBilinearPooling
from CompactBilinearPoolingFourStream import CompactBilinearPooling, CompactBilinearPoolingFourStream
import math

class DI3D(nn.Module):
    #todo backbone sharing weight
    def __init__(self, num_classes, in_channels=3, mode='cat', **kwargs):
        super(DI3D, self).__init__()

        self.backbone_f = InceptionI3d(
            num_classes, in_channels=in_channels, extract_feature=True, **kwargs)
        self.backbone_s = InceptionI3d(
            num_classes, in_channels=in_channels, extract_feature=True, **kwargs)
        self.dropout = nn.Dropout(p=0.5)

        if mode == 'cbp':
            self.mcb = CompactBilinearPooling(1024, 1024, 2048)
            self.fc = nn.Linear(2048, num_classes)

        elif mode == 'cat':
            self.fc = nn.Linear(2048, num_classes)

        else:
            raise NotImplementedError

        self.mode = mode

    def forward(self, x_f, x_s, labels=None):
        feature_f = self.backbone_f(x_f)
        feature_s = self.backbone_s(x_s)

        if self.mode == 'cbp':
            # feature_f = torch.sigmoid(feature_f)
            # feature_s = torch.sigmoid(feature_s)
            # feature_f = self.mcb(feature_f, feature_f)
            # feature_s = self.mcb(feature_s, feature_s)
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

class L2Normalize(nn.Module):
    def __init__(self):
        super(L2Normalize, self).__init__()
        self.var = nn.Parameter(torch.Tensor([4.]))
        self.var.requires_grad = False
        self.num_batches_tracked = 128
        
    def forward(self, feature):
        if self.training:
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            self.var.data = (1.-exponential_average_factor)*self.var.data + exponential_average_factor*torch.norm(feature,p=2,dim=1).mean().detach()
            self.num_batches_tracked += 1
        return feature/self.var

class MBI3D(nn.Module):
    def __init__(self, num_classes, mode='cat', input_modal=['frgb', 'fflow', 'srgb', 'sflow'], share_weight=False, norm='ConstNorm',  **kwargs):
        super(MBI3D, self).__init__()

        self.backbone = dict()
        if share_weight:
            input_modal = [m[1:] for m in input_modal]
        self.input_modal = input_modal
        self.norm = dict()
        self.norm_method = norm
        for m in input_modal:
            if m not in self.backbone:
                if 'rgb' in m:
                    self.backbone[m] = InceptionI3d(
                        num_classes, in_channels=3, extract_feature=True, **kwargs)
                    self.backbone[m].load_state_dict({k: v for k, v in torch.load('models/rgb_imagenet.pt').items()
                                                      if k.find('logits') < 0}, strict=False)

                    if norm == 'L2Norm':
                        self.norm[m] = L2Normalize()
                        self.add_module('norm_%s' % m, self.norm[m])
                    else:
                        self.norm[m] = 1.

                elif 'flow' in m:
                    self.backbone[m] = InceptionI3d(
                        num_classes, in_channels=2, extract_feature=True, **kwargs)
                    self.backbone[m].load_state_dict({k: v for k, v in torch.load('models/flow_imagenet.pt').items()
                                                      if k.find('logits') < 0}, strict=False)
                    if norm == 'L2Norm':
                        self.norm[m] = L2Normalize()
                        self.add_module('norm_%s' % m, self.norm[m])
                    else:
                        self.norm[m] = 4.

                else:
                    raise NotImplementedError
                self.add_module('backbone_%s' % m, self.backbone[m])

        #s = 1.
        #for m in self.norm:
        #    s = s * self.norm[m]
        #self.norm['output'] = math.sqrt(s)

        self.dropout = nn.Dropout(p=0.5)

        if mode == 'cbp':
            if len(input_modal) == 2:
                self.mcb = CompactBilinearPooling(1024, 1024, 2048)
                self.fc = nn.Linear(2048, num_classes)
 
            elif len(input_modal) == 4:
                self.mcb = CompactBilinearPoolingFourStream(
                    1024, 1024, 1024, 1024, 1024*len(input_modal))
                self.fc = nn.Linear(1024*len(input_modal), num_classes)

        elif mode == 'cat':
            self.fc = nn.Linear(1024*len(input_modal), num_classes)

        else:
            raise NotImplementedError

        self.mode = mode
        #self.var = nn.Parameter(torch.Tensor([1.]))
        #self.var.requires_grad = False
        #self.var_lambda = nn.Parameter(torch.Tensor([1e-2]))
        #self.var_lambda.requires_grad = False


    def forward(self, *inputs):

        if self.norm_method == 'L2Norm':
            features = [self.norm[modal](self.backbone[modal](data)) for modal, data in zip(self.input_modal, inputs)]
        else:
            features = [self.backbone[modal](
                data) / self.norm[modal] for modal, data in zip(self.input_modal, inputs)]


        if self.mode == 'cbp':
            feature = self.mcb(*features)
            #if self.training:
            #   print(torch.norm(features[0],p=2,dim=1).mean().detach())
            #    print(torch.norm(features[1],p=2,dim=1).mean().detach())
            #    print(torch.norm(feature,p=2,dim=1).mean().detach())
            #    self.var.data = (1.-self.var_lambda.data)*self.var.data + self.var_lambda.data*torch.norm(feature,p=2,dim=1).mean().detach()
            feature = self.dropout(feature)

        elif self.mode == 'cat':
            feature = torch.cat(features, dim=1)
            feature = self.dropout(feature)
        else:
            raise NotImplementedError

        y = self.fc(feature)
        return y
