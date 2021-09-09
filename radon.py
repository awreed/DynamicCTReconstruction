import torch
from torch import nn
from torch.nn.functional import affine_grid, grid_sample

# 3D version of scikit radon transform
class Radon3D(nn.Module):
    def __init__(self, in_size=None, theta=None, dtype=torch.float):
        super(Radon3D, self).__init__()
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size)

    def forward(self, x):
        N, C, H, W, D = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        N, C, H, W, _ = x.shape
        out = torch.zeros(N, C, H, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1, 1).to(x.device))
            # Sum up the z component
            out[..., i] = rotated.sum(3)

        return out

    def _create_grids(self, angles, grid_size):
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            theta = theta * (3.141592/180)
            # Rotation about the z axis
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0, 0],
                [-theta.sin(), theta.cos(), 0, 0],
                [0,             0,           1, 0]
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, (1, 1, grid_size, grid_size, grid_size)))
        return all_grids
