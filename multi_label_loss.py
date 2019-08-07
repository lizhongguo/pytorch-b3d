import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLL(nn.Module):
    def __init__(self, multi_label_shape):
        super(MLL, self).__init__()
        self._multi_label_shape = tuple(multi_label_shape)
        self._multi_label_idx = 'abcdefghijklm'[:len(self._multi_label_shape)]

        self.criterion = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)
        self.theta = 0.1
        self.weight = [0.1]*len(multi_label_shape)

    def forward(self, output, target):

        # target shape [batch_size, first_label,second_label,... ]
        # first order loss
        output = self.softmax(output)

        output = output.view((-1,)+self._multi_label_shape)
        N = output.size(0)
        assert N == target.size(0)

        tot_loss = None
        for i, idx in enumerate(self._multi_label_idx):
            loss = self.criterion(torch.log(torch.einsum('n%s->n%s' % (self._multi_label_idx, idx), output)),
                                  target[:, i])
            tot_loss = tot_loss + self.weight[i] * \
                loss if tot_loss is not None else loss

        # second order loss
        for i, idx in enumerate(self._multi_label_idx[1:], 1):
            predict = torch.einsum(
                'n%s->na%s' % (self._multi_label_idx, idx), output).view(N, -1)
            loss = self.criterion(torch.log(predict),
                                  self._multi_label_shape[i]*target[:, 0] + target[:, i]) * self.theta

            tot_loss = tot_loss + loss if tot_loss is not None else loss

        return tot_loss

class MLLSampler(object):
    def __init__(self,dataloader,multi_label_shape):
        super(MLLSampler, self).__init__()
        self.dataloader = dataloader
        self._multi_label_shape = tuple(multi_label_shape)
        self._multi_label_idx = 'abcdefghijklm'[:len(self._multi_label_shape)]
    
    def sample_hidden_state(self, model):
        model.train(False)
        with torch.no_grad():
            for _, data in enumerate(self.dataloader):
                inputs, labels, index = data
                inputs = inputs.cuda()
                labels = labels.cuda(non_blocking=True)
                output = model(inputs)

                # target shape [batch_size, first_label,second_label,... ]
                output = F.softmax(output,dim=1)

                output = output.view((-1,)+self._multi_label_shape)

                for i, idx in enumerate(self._multi_label_idx[1:],1):
                    prob = torch.einsum('na%s->na%s' % (self._multi_label_idx[1:], idx), output)
                    self.dataloader.dataset.update_label(index,i,prob)
'''
class MLL_lite(nn.Module):
    def __init__(self, multi_label_shape):
        super(MLL_lite, self).__init__()
        self._multi_label_shape = tuple(multi_label_shape)
        
        idx = 0
        self._multi_label_index = [0]
        for l in self._multi_label_shape[1:]:
            idx = idx + l
            self._multi_label_index.append[idx]


        self.criterion = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)
        self.theta = 0.1
        self.weight = [1.]*len(multi_label_shape)
        self.weight[0] = 1.

    def forward(self, output, target):

        # target shape [batch_size, first_label,(second_label + second_label + ..)]
        # first order loss
        N = output.size(0)

        # N * Action *( L1 + L2 + L3 ...)
        output = output.view((N,self._multi_label_shape[0],-1))

        tot_loss = None

        # second order loss
        for i, idx in enumerate(self._multi_label_shape[1:], 1):
            prob = output[:,:,self._multi_label_index[idx-1]:self._multi_label_index[idx]]
            prob = self.softmax(prob.view(N,-1))
            loss = self.criterion(torch.log(prob),
                self._multi_label_shape[0]*target[:, i] + target[:, 0]) * self.theta

            tot_loss = tot_loss + loss if tot_loss is not None else loss

        return tot_loss
'''