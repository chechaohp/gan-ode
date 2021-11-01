import torch
import torch.nn as nn
from on_dev.mocogan_ode import VideoGeneratorMNIST
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



class VideoGeneratorMNISTODERNN(VideoGeneratorMNIST):
    def __init__(self,n_channels, dim_z_content, dim_z_category, dim_z_motion,
                 video_length, ode_fn=ODEFunc, dim_hidden=None, linear=True,ngf=64):
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

    def sample_z_m(self, num_samples, video_len=None):
        video_len = video_len if video_len is not None else self.video_length

        h_t = [self.get_gru_initial_state(num_samples)]

        for frame_num in range(video_len):
            e_t = self.get_iteration_noise(num_samples)
            h_t_prime = odeint(self.ode_fn, h_t[-1],
                       torch.tensor([0,1]).float())[-1]
            h_t.append(self.recurrent(e_t, h_t_prime))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m