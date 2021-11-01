import torch
import torch.nn as nn
import models.mocogan as mocogan
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()

        self.fn = nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim)
            )

    def forward(self, t, x):
        return self.fn(x)


class VideoGenerator(mocogan.VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=ODEFunc, dim_hidden=None, linear=True,ngf=64):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length,ngf=ngf)
        if dim_hidden:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_hidden)
        else:
            self.ode_fn = ode_fn(dim=dim_z_motion)
        
        if linear:
            self.linear = nn.Sequential(
                    nn.Linear(dim_z_motion, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, dim_z_motion),
                    nn.LeakyReLU(0.2)
                    )
        else:
            self.linear = nn.Identity()

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        
        x = torch.randn(num_samples, self.dim_z_motion)
        if torch.cuda.is_available():
            x = x.cuda()

        x = self.linear(x)

        z_m_t = odeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='rk4')

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t


class VideoGeneratorMNIST(mocogan.VideoGenerator):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=ODEFunc, dim_hidden=None, linear=True,ngf=64):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length,ngf=ngf)
        if dim_hidden:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_hidden)
        else:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_z_motion)
        dim_z = dim_z_motion + dim_z_category + dim_z_content
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, self.n_channels, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

        if linear:
            self.linear = nn.Sequential(
                    nn.Linear(dim_z_motion, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, dim_z_motion),
                    nn.LeakyReLU(0.2)
                    )
        else:
            self.linear = nn.Identity()

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        
        x = torch.randn(num_samples, self.dim_z_motion)
        if torch.cuda.is_available():
            x.cuda()

        x = self.linear(x)

        z_m_t = odeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='rk4')

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t


class VideoGeneratorMNISTODE(VideoGeneratorMNIST):
    def __init__(self, n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=ODEFunc, dim_hidden=None, linear=True,ngf=64):
        super().__init__(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length,ngf=ngf)
        if dim_hidden:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_hidden)
        else:
            self.ode_fn = ode_fn(dim=dim_z_motion, dim_hidden=dim_z_motion)
        
        if linear:
            self.linear = nn.Sequential(
                    nn.Linear(dim_z_motion, 64),
                    nn.LeakyReLU(0.2),
                    nn.Linear(64, dim_z_motion),
                    nn.LeakyReLU(0.2)
                    )
        else:
            self.linear = nn.Identity()

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length
        
        x = torch.randn(num_samples, self.dim_z_motion)
        if torch.cuda.is_available():
            x = x.cuda()

        x = self.linear(x)

        z_m_t = odeint(self.ode_fn, x,
                       torch.linspace(0, 1, video_len).float(),
                       method='rk4')

        z_m_t = z_m_t.transpose(0, 1).reshape(-1, self.dim_z_motion)

        return z_m_t