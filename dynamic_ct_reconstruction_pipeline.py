"""Reconstruction pipeline for method described in https://arxiv.org/abs/2104.11745.
A. W. Reed
Arizona State University
9/10/21"""

import configparser
import constants as C
import numpy as np
import torch
from ct_models import PredictTemplateAndWarp
from losses import WarpFieldTV, SinoL1Loss
import matplotlib.pyplot as plt
from visualization_utils import rotate_camera_movie_volume, get_grid, get_slices
from termcolor import colored

class DynamicCTReconstructionPipeline:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # Optimization spatial dimensions
        self.optim_size = None

        # Axis of rotation for the CT scanner
        self.axis = None

        self.thetas = None
        self.num_proj = None

        # Order of the warp field polynomials
        self.spline_order = None

        # Fourier feature bandwidth
        self.sigma = None

        # Number of features for each input coordinate
        self.nf = None

        # Learning rate of warp field
        self.wf_lr = None
        # Learning rate of the INR
        self.inr_lr = None

        # When to start warp field optimization
        self.wf_start_iter = None
        # How often to increase the resolution of the warp fields
        self.wf_iters_per_resolution = None
        # Coarse-to-fine warp field sizes
        self.wf_sizes = None

        # Number of frames to reconstruct from given angles
        self.num_frames = None
        # Whether to normalize sinogram before computing loss.
        # This makes optimization more stable on the aluminum dataset.
        self.normalize_sino = None

        # How often to save both pipeline output and relevant data.
        self.save_iters = None

        # input sinogram data
        self.input_sino = None

        # normalized size of the voxels
        self.voxel_size = None

        # Number of available GPUs
        self.num_gpu = None
        # name of device
        self.device = None
        # save directory
        self.save_dir = None

        # Shared instance of the reconstruction model
        self.PTW = None

    """Process the input sinogram data and ensure dimensions match what expected"""
    def process_input_sino(self):
        input_sino_path = self.config[C.OPTIMIZATION][C.INPUT_SINO]
        input_size = self.config[C.GEOMETRY].getint(C.INPUT_SIZE)

        self.input_sino = np.fromfile(input_sino_path, dtype=np.float32).reshape(input_size, len(self.thetas), \
                                                               input_size).transpose(0, 2, 1)

        # Input sinogram should be formatted as [1, 1, H, W, num_proj]
        self.input_sino = torch.from_numpy(self.input_sino)[None, None, ...]
        self.input_sino = self.input_sino / self.input_sino.max()

        if input_size != self.optim_size:
            # [H, W, num_proj] -> [num_proj, H, W]
            self.input_sino = self.input_sino.permute(2, 0, 1)[:, None, ...]
            self.input_sino = torch.nn.functional.interpolate(self.input_sino, size=(self.optim_size, self.optim_size),
                                                              mode='bilinear')
            # [num_proj, 1, H, W] -> [1, 1, H, W, num_proj]
            self.input_sino = self.input_sino.squeeze().permute(1, 2, 0)[None, None, ...]

        assert self.input_sino.shape == torch.Size((1, 1, self.optim_size, self.optim_size, self.num_proj)), ("Expected input sino"
                    + "to have dimensions " + "(1, 1, " + str(self.optim_size) + ", " + str(self.optim_size) + ", " +
                    str(self.num_proj) + ") but found " + str(self.input_sino.shape))

        return self.input_sino

    """Process data from the input config file and compute necessary optimization data"""
    def process_config(self):
        self.optim_size = self.config[C.GEOMETRY].getint(C.OPTIM_SIZE)

        self.axis = self.config[C.GEOMETRY][C.AXIS].lower()

        theta_start = self.config[C.GEOMETRY].getint(C.THETA_START)
        theta_stop = self.config[C.GEOMETRY].getint(C.THETA_STOP)
        theta_step = self.config[C.GEOMETRY].getint(C.THETA_STEP)

        self.thetas = torch.arange(theta_start, theta_stop, theta_step)
        self.num_proj = len(self.thetas)

        self.spline_order = self.config[C.OPTIMIZATION].getint(C.SPLINE_ORDER)

        self.sigma = self.config[C.OPTIMIZATION].getfloat(C.SIGMA)
        self.nf = self.config[C.OPTIMIZATION].getint(C.NF)

        self.wf_lr = self.config[C.OPTIMIZATION].getfloat(C.WF_LR)
        self.inr_lr = self.config[C.OPTIMIZATION].getfloat(C.INR_LR)

        self.wf_start_iter = self.config[C.OPTIMIZATION].getint(C.WF_START_ITER)
        self.wf_iters_per_resolution = self.config[C.OPTIMIZATION].getint(C.WF_ITERS_PER_RESOLUTION)

        self.wf_sizes = self.config[C.OPTIMIZATION][C.WF_SIZES]
        self.wf_sizes = self.wf_sizes.split(', ')
        self.wf_sizes = [int(wf.strip()) for wf in self.wf_sizes]

        self.num_frames = self.config[C.OPTIMIZATION].getint(C.NUM_FRAMES)
        self.normalize_sino = self.config[C.OPTIMIZATION].getboolean(C.NORMALIZE_SINO)

        self.save_iters = self.config[C.SAVE_OPTIONS].getint(C.SAVE_ITERS)

        self.save_dir = self.config[C.SAVE_OPTIONS][C.SAVE_DIR]

        self.trans = self.config[C.SAVE_OPTIONS].getboolean(C.TRANS)

        #TODO Update this loader
        self.input_sino = self.process_input_sino()

        self.voxel_size = 2 / self.optim_size
        x_coords = np.linspace(-1 + self.voxel_size / 2, 1 - self.voxel_size / 2, self.optim_size, endpoint=True)
        y_coords = np.linspace(-1 + self.voxel_size / 2, 1 - self.voxel_size / 2, self.optim_size, endpoint=True)
        z_coords = np.linspace(-1 + self.voxel_size / 2, 1 - self.voxel_size / 2, self.optim_size, endpoint=True)
        xyz_grid = np.stack(np.meshgrid(x_coords, y_coords, z_coords), -1)
        # [B, C, X, Y, Z]
        self.xyz_grid = torch.tensor(xyz_grid, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).cuda()

        # [B, C, X, Y, Z] -> [Z, C, X, Y, 1] (so we can parallelize along z dimension)O
        self.xyz_grid = self.xyz_grid.permute(2, 1, 0, 3, 4)

    """Detect if using a GPU. Good luck running if on CPU"""
    def detect_gpu(self):
        if torch.cuda.is_available():
            self.num_gpu = torch.cuda.device_count()
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    """Save movie at epoch, network and warp field parameters, and data necessary to upsample in time and space after
    optimization"""
    def save_data(self, CTData, epoch):
        movie = CTData.movie.squeeze().detach().cpu().numpy()
        template = CTData.template.squeeze().detach().cpu().numpy()

        plt.clf()
        plt.imsave(self.save_dir + 'slices_' + str(epoch) + '.png', get_grid(get_slices(movie,
                                                                                        dim=self.optim_size // 2,
                                                                                        axis=2), cols=5))
        rotate_camera_movie_volume(movie.copy(), self.save_dir + 'static_' + str(epoch) + '.mp4', rotate=None,
                                   normalize=True, color='cool', fps=2, trans=self.trans)

        np.save(self.save_dir + 'movie_npy' + str(epoch) + '.npy', movie)
        np.save(self.save_dir + 'template_npy' + str(epoch) + '.npy', template)

        wf_coeffs = CTData.wf_coeffs.detach().cpu().numpy()
        size = wf_coeffs.shape[-1]
        np.save(self.save_dir + 'wf_coeffs_' + str(size) + '.npy', wf_coeffs)

        torch.save(self.PTW.INR.module.state_dict(), self.save_dir + 'network_' + str(epoch) + '.pth')

    """Run the optimization"""
    def run_pipeline(self):
        print(colored("Processing config file...", 'green'))
        self.process_config()

        self.detect_gpu()

        self.PTW = PredictTemplateAndWarp(self).to(self.device)

        self.input_sino = self.input_sino.to(self.device)

        TV_Loss = WarpFieldTV()
        L1_Loss = SinoL1Loss()

        optimizer = torch.optim.Adam([
            {'params': self.PTW.WF_Model.parameters(), 'lr': self.wf_lr},
            {'params': self.PTW.INR.parameters(), 'lr': self.inr_lr}])

        num_epochs = self.wf_start_iter + len(self.wf_sizes)*self.wf_iters_per_resolution

        print(colored("Starting optimization...", 'green'))
        for epoch in range(1, num_epochs):
            optimizer.zero_grad()

            CTData = self.PTW(self.xyz_grid, epoch, perturb=False)

            if self.normalize_sino:
                CTData.est_sino = CTData.est_sino / CTData.est_sino.max().detach()

            # Retrieve batch of projections from ground truth sinogram
            gt_sino_batches = self.input_sino[..., self.PTW.WF_Model.theta_batches]

            assert gt_sino_batches.shape == (1, 1, self.optim_size, self.optim_size, self.num_frames)
            assert CTData.est_sino.shape == (1, 1, self.optim_size, self.optim_size, self.num_frames)

            l1_loss = L1_Loss(gt_sino_batches, CTData.est_sino, weight=1.0)
            tv_loss = TV_Loss(CTData.wf_coeffs, normalize=True, weight=0.1)

            total_loss = l1_loss + tv_loss

            total_loss.backward()

            optimizer.step()

            print(colored(''.join(["Epoch: ", str(epoch), " | ", "Optimization Loss: ",
                                   str(total_loss.item())]), 'blue'))

            if epoch % self.save_iters == 0:
                print(colored("Saving...", 'green'))
                self.save_data(CTData, epoch)





