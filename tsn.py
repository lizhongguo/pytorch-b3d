import torch
import inceptionv3
import torch.nn as nn
import torch.nn.functional as F
class TSN(nn.Module):

    def __init__(self, num_classes, in_channels=3, **kwargs):
        super(TSN, self).__init__()
        self.backbone = inceptionv3.inception_v3(pretrained=True, aux_logits=False, **kwargs)
        self.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax(dim = 1)
        self.num_classes = num_classes
        if in_channels == 2:
            self.backbone.Conv2d_1a_3x3 = inceptionv3.BasicConv2d(2, 32, kernel_size=3, stride=2)

    def forward(self, x):
        """forward 
        
        Args:
            x (Tensor): Shape N C T H W
        """
        x = x.permute(0,2,1,3,4)
        shape = list(x.shape)
        x = x.reshape([-1]+shape[2:])
        # x Shape N*T CHW
        x = self.backbone(x)
        # x Shape N*T 2048
        x = self.fc(x)
        x = self.softmax(x)
        x = x.reshape(shape[:2]+[self.num_classes])
        x = x.mean(dim = 1, keepdim=True)
        if self.training:
            x = torch.log(x)
        #x = x.max(dim = 1, keepdim=True)
        x = x.squeeze(dim = 1)
        return x
