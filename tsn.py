import torch
import inceptionv3
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class TSN(nn.Module):

    def __init__(self, num_classes, in_channels=3, **kwargs):
        super(TSN, self).__init__()
        #self.base_model = inceptionv3.inception_v3(pretrained=True, aux_logits=False, **kwargs)
        self.base_model = pretrainedmodels.bninception()

        if in_channels == 3:
            state_dict = torch.load('models/kinetics_rgb.pth')
        elif in_channels == 2:
            state_dict = torch.load('models/kinetics_flow.pth')

        state_dict = {k.rsplit('base_model.', 1)[1]: v for k,
                    v in state_dict.items() if 'fc_action' not in k}
        self.base_model.load_state_dict(state_dict, strict=False)

        self.num_classes = num_classes
        self._enable_pbn = True
        self._prepare_tsn()

        if in_channels == 2:
            self._construct_flow_model()

    def _prepare_tsn(self):
        feature_dim = self.base_model.last_linear.in_features
        self.base_model.last_linear = nn.Sequential(nn.Dropout(p=0.8),
                                                    nn.Linear(feature_dim, self.num_classes))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            #print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def _construct_flow_model(self):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(
            modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        self.new_length = 5
        new_kernel_size = kernel_size[:1] + \
            (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(
            dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        # remove .weight suffix to get the layer name
        layer_name = list(container.state_dict().keys())[0][:-7]

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def forward(self, x):
        """forward 

        Args:
            x (Tensor): Shape N C T H W
        """
        x = x.permute(0, 2, 1, 3, 4)
        shape = list(x.shape)
        x = x.reshape([-1]+shape[2:])
        # x Shape N*T CHW
        x = self.base_model(x)
        # x Shape N*T 2048
        #x = self.softmax(x)
        x = x.reshape(shape[:2]+[self.num_classes])
        x = x.mean(dim=1, keepdim=True)
        # if self.training:
        #    x = torch.log(x)
        #x = x.max(dim = 1, keepdim=True)
        x = x.squeeze(dim=1)
        return x
