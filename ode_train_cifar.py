"""
Modified from https://github.com/pytorch/examples/blob/master/dcgan/main.py
"""
from __future__ import print_function
import argparse
import os
import random
import copy
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from on_dev.ode_training import GANODETrainer
from torch.utils.tensorboard import SummaryWriter

dataset = 'cifar10'
dataroot = '../data'
workers = 8
batchSize = 64
imageSize = 32
nz = 128
ngf = 64
ndf = 64
niter = 250
cuda = torch.cuda.is_available()
ngpu = 1
netG = ''
netD = ''
outf = '../images'
ode = 'rk4'
step_size = 0.01
disc_reg = 0.01
manualSeed = None
dry_run = False # to run test


outf = os.path.join(outf, dataset + "_" + ode)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logdir = os.path.join(outf, 'logs', timestamp)

writer = SummaryWriter(log_dir=logdir)

try:
    os.makedirs(outf, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
except OSError:
    pass

if manualSeed is None:
    manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if dataroot is None and str(dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % dataset)

if dataset == 'cifar10':
    dataset = dset.CIFAR10(root=dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    nc = 3

elif dataset == 'mnist':
    dataset = dset.MNIST(root=dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]))
    nc = 1


assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                            shuffle=True, num_workers=int(workers))

device = torch.device("cuda:0" if cuda else "cpu")

# Conv Initialization from SNGAN codebase
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.project = nn.Conv2d(nz, ngf * 8 * 4 * 4, 1, 1, 0, bias=False)
        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            raise NotImplemented()
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            x = self.project(input)
            x = x.view(-1, ngf * 8, 4, 4)
            output = self.main(x)

        return output


# ODE GAN
netG = Generator(ngpu)
netG.apply(weights_init)
netG = netG.to(device)

if netG != '':
    netG.load_state_dict(torch.load(netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)

netD = netD.to(device)

if netD != '':
    netD.load_state_dict(torch.load(netD))
print(netD)

criterion = nn.BCEWithLogitsLoss()

# fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

if dry_run:
    niter = 1

# deep copies model + grads of model
def grad_clone(source: torch.nn.Module) -> torch.nn.Module:
    dest = copy.deepcopy(source)
    dest.requires_grad_(True)

    for s_p, d_p in zip(source.parameters(), dest.parameters()):
        if s_p.grad is not None:
            d_p.grad = s_p.grad.clone()

    return dest

# Inplace normalizes gradient; if grad_norm > 1
def normalize_grad(grad: torch.Tensor) -> torch.Tensor:
    # normalize gradient
    grad_norm = grad.norm()
    if grad_norm > 1.:
        grad.div_(grad_norm)
    return grad



def gan_step(G: Generator, D: Discriminator, data, detach_err: bool = True, retain_graph: bool = False):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    D.zero_grad()

    real_cpu = data[0].to(device)
    batch_size = real_cpu.size(0)
    label = torch.full((batch_size,), real_label,
                        dtype=real_cpu.dtype, device=device)

    output = D(real_cpu)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().detach()

    # train with fake
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = G(noise)
    label.fill_(fake_label)
    output = D(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().detach()
    errD = errD_real + errD_fake

    if detach_err:
        errD = errD.detach()

    DISC_GRAD_CACHE = grad_clone(D)

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    G.zero_grad()

    label.fill_(real_label)  # fake labels are real for generator cost
    output = D(fake)
    errG = criterion(output, label)
    errG.backward(create_graph=retain_graph)
    D_G_z2 = output.mean().detach()

    if detach_err:
        errG = errG.detach()

    GEN_GRAD_CACHE = grad_clone(G)

    return DISC_GRAD_CACHE, GEN_GRAD_CACHE, errD, errG, D_x, D_G_z1, D_G_z2

# Save hyper parameters
# writer.add_hparams(vars(opt), metric_dict={})

step_size = step_size
global_step = 0
trainer = GANODETrainer(netG.parameters(), netD.parameters(), None, g_loss, d_loss, None, step_size)

for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        # Schedule
        if global_step < 500:
            step_size = step_size
        elif global_step >= 500 and global_step <= 400000:
            step_size = step_size * 4
        elif global_step > 400000:
            step_size = step_size * 2


        elif ode == 'rk4':
            netG, netD, errD, errG, D_x, D_G_z1, D_G_z2, gen_grad_norm, disc_grad_norm = rk4_ode_step(netG, netD,
                                                                                                        data,
                                                                                                        step_size=step_size,
                                                                                                        disc_reg=disc_reg)

        else:
            raise ValueError("Only support ode steps are - heun and rk4")

        # Cast logits to sigmoid probabilities
        D_x = D_x.sigmoid().item()
        D_G_z1 = D_G_z1.sigmoid().item()
        D_G_z2 = D_G_z2.sigmoid().item()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f '
                'Gen Grad Norm: %0.4f Disc Grad Norm: %0.4f'
                % (epoch, niter, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, gen_grad_norm, disc_grad_norm))

        writer.add_scalar('loss/discriminator', errD.item(), global_step=global_step)
        writer.add_scalar('loss/generator', errG.item(), global_step=global_step)
        writer.add_scalar('acc/D(x)', D_x, global_step=global_step)
        writer.add_scalar('acc/D(G(z))-fake', D_G_z1, global_step=global_step)
        writer.add_scalar('acc/D(G(z))-real', D_G_z2, global_step=global_step)
        writer.add_scalar('norm/gen_grad_norm', gen_grad_norm, global_step=global_step)
        writer.add_scalar('norm/disc_grad_norm', disc_grad_norm, global_step=global_step)
        writer.add_scalar('step_size', step_size, global_step=global_step)

        global_step += 1

        if i % 100 == 0:
            real_cpu = data[0].to(device)
            vutils.save_image(real_cpu,
                                '%s/real_samples.png' % outf,
                                normalize=True)

            # fake = netG(fixed_noise)
            random_noise = torch.randn(batchSize, nz, 1, 1, device=device)

            fake = netG(random_noise)

            vutils.save_image(fake.detach(),
                                '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                                normalize=True)

        if dry_run:
            break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

writer.flush()