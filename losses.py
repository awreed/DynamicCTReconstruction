import torch
import torch.nn as nn

"""Total variation loss applied to the warp field"""


class WarpFieldTV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wf, normalize=False, weight=0.1):
        [_, _, x, y, z] = wf.shape
        # [Order, C, X, Y, Z]
        tv = torch.sum(torch.abs(wf[..., 1:, :, :] - wf[..., :-1, :, :])) / x + \
             torch.sum(torch.abs(wf[..., :, 1:, :] - wf[..., :, :-1, :])) / y + \
             torch.sum(torch.abs(wf[..., :, :, 1:] - wf[..., :, :, :-1])) / z

        # normalize magnitude by dimension of warp field for smoother loss landscape
        if normalize:
            tv = tv / wf.numel() ** (1 / 2)

        tv = weight * tv

        return tv


"""L1 Loss applied in sinogram space"""


class SinoL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt, est, weight=1.0):
        return torch.nn.functional.l1_loss(gt, est) * weight
