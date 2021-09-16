import torch
import torch.nn as nn

from models.mocogan.torch_bnn import BNN

import numpy as np

class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length, ngf=63):
        super().__init__()
        # input channel
        self.n_channels = n_channels
        # latent content dimension
        self.dim_z_content = dim_z_content
        # latent category dimension
        self.dim_z_category = dim_z_category
        # latent motion dimension
        self.dim_z_motion = dim_z_motion
        # video length dimension
        self.video_length = video_length
        # latent dimension
        dim_z = dim_z_motion + dim_z_category + dim_z_content
        # bayesian network, this network will govern the ode function
        # TODO: add bayesian network here
        self.bnn = BNN  

        # main network
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def sample_z_m(self, num_samples, video_len = None):
        """
        Sample motion vector
        """
        pass

    def sample_z_category(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)
