import torch
import torch.nn as nn
import torchcde
import on_dev.mocogan as mocogan
import numpy as np
from torch.autograd import Variable

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

class CDEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(hidden_dim, 128)
        self.linear2 = nn.Linear(128, input_dim * hidden_dim)

    def forward(self, t, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)

        z = torch.tanh(z)

        z = z.view(z.size(0), self.hidden_dim, self.input_dim)
        return z


class VideoGenerator(mocogan.VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf=64, ode_func = CDEFunc):
        super(VideoGenerator, self).__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ngf)

        self.ode_func = ode_func(dim_z_motion, dim_z_motion)

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        # generate initial Z
        z_0 = self.get_gru_initial_state(num_samples)
        # generate X
        X_t = self.get_iteration_noise(num_samples)
        # generate time
        t = torch.linspace(0,1, video_len)
        # repeat for sum_samples
        t = t.unsqueeze(0).repeat(128,1)
        # concatenate X with time
        X_t = torch.stack([t,X_t])

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m



