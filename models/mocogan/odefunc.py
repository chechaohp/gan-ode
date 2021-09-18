import torch.nn as nn

class SimpleODEFunc(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,dim)
        )
    
    def forward(self, t, x):
        return self.main(x)