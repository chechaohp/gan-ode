# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from on_dev.ode_training import GANODETrainer
from tqdm import tqdm

# set seed
# torch.manual_seed(0)
import random
# random.seed(0)
import numpy as np
# np.random.seed(0)

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))
    
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def d_loss(x):
    # Discriminator
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    # D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    # D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    return D_loss


def g_loss():
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)
    return G_loss


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set batch size
bs = 100

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# build network
z_dim = 10
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)

# loss
criterion = nn.BCELoss() 
n_epoch = 100
# optimizer
# lr = 0.0002 
# G_optimizer = optim.Adam(G.parameters(), lr = lr)
# D_optimizer = optim.Adam(D.parameters(), lr = lr)
method='rk4'
ode_trainer = GANODETrainer(G.parameters(), D.parameters(), None, g_loss, d_loss, None,method=method)

d_iter = 2
g_iter = 1

d_losses = []
g_losses = []
fixed_test_z = torch.randn(bs,z_dim).to(device)

for epoch in tqdm(range(1,n_epoch+1),desc='Training process'):
    for batch_idx, (x, _) in enumerate(train_loader):
        for i in range(d_iter):
            disLoss = ode_trainer.step(x,model='dis_img')
            d_losses.append(disLoss.item())
    
        for i in range(g_iter):
            genLoss = ode_trainer.step(model='gen')
            g_losses.append(genLoss)
        
    if (epoch % 10) == 0:
        print(f'[EPOCH {epoch}/{n_epoch}] - Gen Loss {genLoss.item()} disLoss {disLoss.item()}')
        with torch.no_grad():
            generated = G(fixed_test_z)
            save_image(generated.view(generated.size(0), 1, 28, 28), f'./sample_{method}_{epoch}.png')