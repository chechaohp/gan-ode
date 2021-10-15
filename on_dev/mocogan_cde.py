import torch
import torch.nn as nn
import torchcde
from on_dev.mocogan_ode import VideoGeneratorMNIST
# import numpy as np
# from torch.autograd import Variable

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

class CDEFunc(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.input_dim = dim
        self.hidden_dim = dim_hidden

        self.linear1 = nn.Linear(dim_hidden, 128)
        self.linear2 = nn.Linear(128, dim * dim_hidden)

    def forward(self, t, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)

        z = torch.tanh(z)

        z = z.view(z.size(0), self.hidden_dim, self.input_dim)
        return z


class VideoGeneratorCDE(VideoGeneratorMNIST):
    def __init__(self,n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=CDEFunc, dim_hidden=None, linear=True,ngf=64):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length,ngf=ngf)
        if dim_hidden:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_hidden)
        else:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden = dim_z_motion)
        
        if linear:
            self.linear = nn.Sequential(
                    nn.Linear(dim_z_motion, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, dim_z_motion),
                    nn.LeakyReLU(0.2)
                    )
        else:
            self.linear = nn.Identity()

        self.f = nn.Sequential(
                    nn.Linear(dim_z_motion, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, dim_z_motion),
                    nn.LeakyReLU(0.2)
                    )

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        # generate initial Z
        # z_0 = self.get_gru_initial_state(num_samples)
        # generate X
        X_t = self.get_iteration_noise(num_samples)
        batch, z_dim = X_t.size()
        t = torch.linspace(0,1,video_len)
        t_ = t.repeat(batch).view(batch,video_len,1)
        x_ = X_t.view(batch,video_len,1)
        x = torch.cat([t_,x_],dim=2)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
        X = torchcde.CubicSpline(coeffs)
        z0 = self.f(X.evaluate(X.interval[0]))
        z_T = torchcde.cdeint(X=X,z0=z0,func=cde_fn,t=torch.arange(0,video_len).float())

        z_T = z_T.view(-1,video_len)

        return z_T



