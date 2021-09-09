import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import affine_grid, grid_sample
from collections import namedtuple
from nn_models import INR_3D, FourierFeatures
from termcolor import colored

"""Wrapper for grid_sample so we can parallelize the op"""


class GridSampleParallel(nn.Module):
    def __init__(self):
        super(GridSampleParallel, self).__init__()

    def forward(self, frames, wf):
        return F.grid_sample(frames, wf.permute(0, 2, 3, 4, 1), align_corners=False)


"""This class simulates the sinogram for our forward CT model. It assumes the reconstructed sinogram contains a row
for the projection taken at each angle. For example, for the 4D sinogram for 90 projections will assume those 90
projections come from 90 linear time steps."""


class CT4DForwardSimulation(nn.Module):
    def __init__(self, opt):
        super(CT4DForwardSimulation, self).__init__()
        self.thetas = opt.thetas
        self.axis = opt.axis
        self.optim_size = opt.optim_size

        # precompute the affine grids for the radon transform at each angle
        self.all_grids = self._create_grids(self.thetas, axis=self.axis)

    def forward(self, x, theta_batches):
        _, C, H, W, _ = x.shape

        out_sino = torch.zeros(1, C, H, W, len(theta_batches)).to(x.device)

        # sample the indeces that we are randomly sampling on this batch
        for i, index in enumerate(theta_batches):
            # Rotate about z axis and sum along depth dimension to simulate forward CT process
            out_sino[..., i] = grid_sample(x[i, ...].unsqueeze(0), self.all_grids[index].to(x.device),
                                           align_corners=True, mode='bilinear').sum(3)

        return out_sino

    # Create affine grids for rotating scene (simulate rotating platform)
    def _create_grids(self, angles, axis='z'):
        all_grids = []
        for theta in angles:
            # convert to radians
            theta = theta * (3.141592 / 180)
            # Rotation about z axis
            if axis == 'x':
                R = torch.tensor([[
                    [1, 0, 0, 0],
                    [0, theta.cos(), theta.sin(), 0],
                    [0, -theta.sin(), theta.cos(), 0]
                ]])
            # Rotation about y axis
            elif axis == 'y':
                R = torch.tensor([[
                    [theta.cos(), 0, -theta.sin(), 0],
                    [0, 1, 0, 0],
                    [theta.sin(), 0, theta.cos(), 0]
                ]])
            else:  # else rotate around the z axis
                R = torch.tensor([[
                    [theta.cos(), theta.sin(), 0, 0],
                    [-theta.sin(), theta.cos(), 0, 0],
                    [0, 0, 1, 0]
                ]])
            all_grids.append(affine_grid(R, (1, 1, self.optim_size, self.optim_size, self.optim_size),
                                         align_corners=True))
        return all_grids


"""This class handles updating and upsampling warp field parameters throughout the optimization."""


class Coarse2FineWarpField(nn.Module):
    def __init__(self, opt):
        super(Coarse2FineWarpField, self).__init__()

        self.sizes = opt.wf_sizes
        self.coarse_2_fine = nn.ParameterList()
        self.spline_order = opt.spline_order
        self.num_proj = opt.num_proj
        self.num_frames = opt.num_frames
        self.wf_start_iter = opt.wf_start_iter
        self.wf_iters_per_resolution = opt.wf_iters_per_resolution
        self.optim_size = opt.optim_size
        self.device = opt.device

        # Add the warp field parameters at each size to the a list
        for size in self.sizes:
            self.coarse_2_fine.append(
                nn.Parameter(torch.zeros((self.spline_order + 1, 3, size, size, size))))

        self.res = -1

        # Time vector to create warp field
        self.times = np.linspace(0, 1, opt.num_proj, endpoint=False)
        self.times = torch.from_numpy(self.times)
        self.times = self.times.type(torch.float32).to(self.device)
        # self.times = self.times[..., None, None, None, None].cuda()

        self.delta_t = (self.times[1] - self.times[0])

        for p in self.parameters():  # Turn off the warp field parameters initially
            p.requires_grad_(False)

        self._count = 0  # internal counter for the warp field

        self.theta_batches = None  # gets set

    def _get_random_times(self):
        # Get random time centroids
        batch_times_indeces = np.random.choice(self.num_proj, size=self.num_frames)

        # Get random perturbation of time
        batch_time_offsets = -self.delta_t * torch.randn(self.num_frames).cuda() + self.delta_t / 2

        # Add the random perturbation and clamp the output to between [0, 1]
        batch_times = torch.clamp(self.times[batch_times_indeces] + batch_time_offsets, 0, self.times[-1])

        # sort the times in ascending order -- easier to debug movie
        batch_times, indeces = torch.sort(batch_times)

        # Convert the time to angle according to our sampling scheme
        self.theta_batches = torch.round(batch_times / self.delta_t).long()

        # Return as [times, 1, 1, 1, 1] to be multiplied by warp field coefficients
        return batch_times[:, None, None, None, None]

    def _check_update(self, epoch):
        if epoch == self.wf_start_iter:
            for p in self.parameters():  # Turn on the warp field gradients
                p.requires_grad_(True)

        if epoch >= self.wf_start_iter:  # If the warp fields are turned on
            # If we need to upsample
            if self._count % self.wf_iters_per_resolution == 0 and self.res < len(self.sizes) - 1:
                self.res = self.res + 1
                print(colored(' '.join(["Upsampling warp fields to", str(self.sizes[self.res]), "X",
                                        str(self.sizes[self.res]), "X", str(self.sizes[self.res])]), 'green'))
                # initialize the upsampled warp field to the interpolated values from previous size
                self.coarse_2_fine[self.res].data = torch.nn.functional.interpolate(self.coarse_2_fine[self.res - 1],
                                                                                    size=(self.sizes[self.res],
                                                                                          self.sizes[self.res],
                                                                                          self.sizes[self.res]),
                                                                                    mode='trilinear',
                                                                                    align_corners=False)
            self._count = self._count + 1

    def forward(self, epoch):
        # Check if we need to upsample the warp field
        self._check_update(epoch)

        # Sample warp field polynomial at times
        wf_coeffs = self.coarse_2_fine[self.res]

        diff_wf = 0

        # Get randomly sorted times and set theta_batches for when we take the sinogram
        batch_times = self._get_random_times()

        # calculate the warp field polynomial
        for order in range(self.spline_order + 1):
            diff_wf += wf_coeffs[order] * (batch_times ** (order))

        # Interpolate warp field up to full image resolution
        return torch.nn.functional.interpolate(diff_wf, size=(self.optim_size, self.optim_size,
                                                              self.optim_size),
                                               mode='trilinear', align_corners=False), wf_coeffs


"""This is the main class for running the optimization. It predicts a template object, warps the template, 
computes its sinogram and returns relevant information to the main optimization loop."""


class PredictTemplateAndWarp(nn.Module):
    def __init__(self, opt):
        super(PredictTemplateAndWarp, self).__init__()
        self.device = opt.device
        self.axis = opt.axis
        self.sigma = opt.sigma
        self.nf = opt.nf
        self.voxel_size = opt.voxel_size
        self.optim_size = opt.optim_size
        self.num_frames = opt.num_frames

        # NN to predict the first frame of movie
        self.INR = nn.DataParallel(INR_3D(opt)).to(self.device)
        self.GaussianFeatureExtractor = nn.DataParallel(FourierFeatures(opt)).to(self.device)

        self.ApplyWarpField = nn.DataParallel(GridSampleParallel()).to(self.device)

        # Counter for resolution of warp field
        self.WF_Model = Coarse2FineWarpField(opt).to(self.device)

        # Hard clipping to keep warp field in bounds of scene
        self.Clipper = nn.Hardtanh(-1, 1)

        # Create estimate sinogram by extracting relevant rows from each frame's sinogram
        self.SimulateForwardCT = CT4DForwardSimulation(opt).to(self.device)

        self.CTData = namedtuple('CTData', ['template', 'movie', 'est_sino', 'wf_coeffs'])

    def forward(self, xyz_grid, epoch, perturb=False):
        # Perturb the input coordinates?
        if perturb:
            xyz_pert = self.voxel_size / 2 * torch.randn([self.optim_size, 1, 1, self.optim_size,
                                                          self.optim_size],
                                                         dtype=torch.float32)
            xyz_pert = xyz_pert.float()
            xyz_grid = xyz_grid + xyz_pert.to(xyz_grid.device)

        # Compute fourier features for input coordinates
        xyz_ff = self.GaussianFeatureExtractor(xyz_grid)

        # Estimate the static image [X, 1, 1, Y, Z] -> [1, 1, X, Y, Z]
        template = self.INR(xyz_ff).permute(1, 2, 0, 3, 4)
        # Copy the image across the batch dimension
        template_copy = template.repeat(self.num_frames, 1, 1, 1, 1)

        # Calculate the spline fitted warp field
        diff_wf, wf_coeffs = self.WF_Model(epoch)

        # Clip warp field - permute xyz_grid back to [1, 1, X, Y, Z]
        wf = self.Clipper(xyz_grid.permute(2, 1, 0, 3, 4).to(diff_wf.device) + diff_wf)
        # Sample each frame using the spline warp field
        movie = self.ApplyWarpField(template_copy, wf)
        # Create estimate sinogram by taking relevant projections from each frame of ct_movie
        est_sino = self.SimulateForwardCT(movie, self.WF_Model.theta_batches)

        # Return the data to optimization loop
        self.CTData.template = template
        self.CTData.movie = movie
        self.CTData.est_sino = est_sino
        self.CTData.wf_coeffs = wf_coeffs

        return self.CTData
