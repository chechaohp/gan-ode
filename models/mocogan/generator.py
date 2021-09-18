from os import replace
from numpy.lib.npyio import zipfile_factory
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.mocogan.bnn import BNN
from models.mocogan.odefunc import SimpleODEFunc

from torchdiffeq import odeint

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
        self.bnn = BNN(dim_z_motion, dim_z_motion, n_hid_layers=2, act='softplus',layer_norm=True,bnn=True)

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
    
    def get_ode_func(self):
        """ get ODE function
        """
        # TODO:get ODE function
        f = SimpleODEFunc(self.dim_z_motion, self.dim_z_motion * 2)
        return f

    def sample_z_m(self, num_samples, video_len = None):
        """
        Sample motion vector
        """
        # TODO use ODE function to get all z motion
        video_len = video_len if video_len is not None else self.video_length
        # create random initial state
        z0 = torch.randn(num_samples, self.dim_z_motion)
        # get the ode function
        ode_func = self.get_ode_func()
        # use ode to generate the sequence of motion latent
        z_m_t = odeint(ode_func, z0, torch.linspace(0,1,video_len).float(), method='rk4')
        z_m_t = z_m_t.transpose(0,1).reshape(-1, self.dim_z_motion)
        return z_m_t

    def sample_z_categ(self, num_samples, video_len=None):
        """ Sample the class to generate
        """
        # get the video len
        video_len = video_len if video_len is not None else self.video_length
        # when the the category dimension is less or equal 0, this is unconditional gan
        if self.dim_z_category <= 0:
            return None, np.zeros(num_samples)
        # random the class of fake video
        classes_to_generate = np.random.ranint(self.dim_z_category, size=num_samples)
        # create one hot vector for class
        one_hot = np.zeros((num_samples, self.dim_z_category), dtype=float)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        # repeat it to input generator
        one_hot_video = np.repeat(one_hot, video_len,axis=0)
        # convert to pytorch
        one_hot_video = torch.from_numpy(one_hot_video)

        if torch.cuda.is_available():
            one_hot_video = one_hot_video.cuda()

        return Variable(one_hot_video), classes_to_generate


    def sample_z_content(self, num_samples, video_len=None):
        """ Sample latent vector for content
        """
        # get video len
        video_len = video_len if video_len is not None else self.video_length
        # get random content vector
        content = np.random.normal(0, 1, (num_samples, self.dim_z_content), dtype=float)
        # repeat content vector for video
        content = np.repeat(content, video_len, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return Variable(content)


    def sample_z_video(self, num_samples, video_len=None):
        """ Generate latent z
        """
        z_content = self.sample_z_content(num_samples, video_len)
        z_category, z_category_labels = self.sample_z_categ(num_samples, video_len)
        z_motion = self.sample_z_m(num_samples, video_len)

        if z_category is not None:
            z = torch.cat([z_content, z_category, z_motion],dim=1)
        else:
            z = torch.cat([z_content, z_motion], dim=1)
        return z, z_category_labels


    def sample_videos(self, num_samples, video_len=None):
        """ Generate fake videos
        """
        video_len = video_len if video_len is not None else self.video_length
        z, z_category_labels = self.sample_z_video(num_samples, video_len)
        h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        h = h.view(h.size(0) / video_len, video_len, self.n_channels, h.size(3), h.size(3))

        z_category_labels = torch.from_numpy(z_category_labels)

        if torch.cuda.is_available():
            z_category_labels = z_category_labels.cuda()
        
        h = h.permute(0,2,1,3,4)
        return h, Variable(z_category_labels, requires_grad=False)

    
    def sample_images(self, num_samples):
        z, z_category_labels = self.sample_z_video(num_samples * self.video_length * 2)

        j = np.sort(np.random.choice(z.size(0), num_samples, replace=False)).astype(int)
        z = z[j,::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, None
