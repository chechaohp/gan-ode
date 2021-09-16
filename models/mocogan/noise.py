import torch
import torch.nn as nn

class Noise(nn.Module):
    def __init__(self, use_noise, device, sigma=0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma
        self.dev = device

    def forward(self,x):
        if self.use_noise:
            noise = self.sigma * torch.randn_like(x).to(self.device)
            return x + noise

