from DWT import DWT3D
import torch
x = torch.randn(1,3,2,2,2)
x.requires_grad=True

dwt = DWT3D('haar')

s = dwt(x)

s = s.reshape(3,8)
s = s[:,1].sum()
s.backward()

dwt = DWT3D('haar',only_hw=True)
s = dwt(x)

s = s.reshape(3,4,2)
s = s[:,1,:].sum()
s.backward()

print(x.grad)
