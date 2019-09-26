import torch
from tensorboardX import SummaryWriter
import torchvision
logger = SummaryWriter(comment='feature_vis')
feature = torch.load('pev_feature_bpi3d_rgb_result.pt', map_location=lambda storage, loc: storage)
for k, v in feature.items():

    min = v.min()
    max = v.max()
    v.clamp_(min=min, max=max)
    v.add_(-min).div_(max - min + 1e-5)

    imgs = v.permute(1, 0, 2, 3).reshape(-1, 1, 14, 14)

    logger.add_image('feature/img', torchvision.utils.make_grid(imgs, nrow=8), k.item())

    v = v.view(64,8,1,14,14)
    logger.add_video('feature/video', v, k.item())