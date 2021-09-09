import torch
import torch.nn as nn
import numpy as np

"""Computes fourier features for the INR"""


class FourierFeatures(nn.Module):
    def __init__(self, opt):
        super().__init__()
        torch.manual_seed(0)
        self.num_input_channels = 3
        self.num_features = opt.nf
        self.sigma = opt.sigma
        self.B = (torch.randn((self.num_input_channels, self.num_features), dtype=torch.float32) * self.sigma)

    def forward(self, x):
        assert x.dim() == 5, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height, depth = x.shape

        assert channels == self.num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self.num_input_channels, channels)
        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 4, 1).reshape(batches * width * height * depth, channels)
        x = x.type(torch.float32)

        x = x @ self.B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, depth, self.num_features)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 4, 1, 2, 3)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


"""Swish activation function"""


class Swish(nn.Module):
    def forward(self, input):
        return input * torch.sigmoid(input)


"""INR for reconstructing the scene template"""


class INR_3D(nn.Module):
    def __init__(self, opt):
        super(INR_3D, self).__init__()
        self.nf = opt.nf

        self.main = nn.Sequential(
            nn.Conv3d(self.nf * 2, self.nf * 2, kernel_size=1, padding=0),
            Swish(),
            nn.Conv3d(self.nf * 2, self.nf * 2, kernel_size=1, padding=0),
            Swish(),
            nn.Conv3d(self.nf * 2, self.nf * 2, kernel_size=1, padding=0),
            Swish(),
            nn.Conv3d(self.nf * 2, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
