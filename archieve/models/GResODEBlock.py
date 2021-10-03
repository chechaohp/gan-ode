import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torchdiffeq import odeint
from models.Normalization import ConditionalNorm, l2normalize

class SpectralNorm(nn.Module):
    """ Modify from Normalization.SpectralNorm to work with ODE function
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module.layer, self.name + "_u")
        v = getattr(self.module.layer, self.name + "_v")
        w = getattr(self.module.layer, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module.layer, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module.layer, self.name + "_u")
            v = getattr(self.module.layer, self.name + "_v")
            w = getattr(self.module.layer, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module.layer, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module.layer._parameters[self.name]

        self.module.layer.register_parameter(self.name + "_u", u)
        self.module.layer.register_parameter(self.name + "_v", v)
        self.module.layer.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Conv2dODE(nn.Module):
    """ Use for ODE function
    """
    def __init__(self, in_channel, out_channel, ksize=3, stride=1, 
                 padding=0, bias=True):
        super().__init__()
        # for augmented
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer = nn.Conv2d(out_channel,out_channel,ksize,stride,padding,bias = bias)

    def forward(self, t, x):
        # BT, C, W, H = x.size()
        # zeros augmented
        # if self.in_channel < self.out_channel:
        #     zeros_aug = torch.zeros([BT, self.out_channel - self.in_channel, W, H])
        #     x = torch.cat((x,zeros_aug),1)
        x = x * t
        return self.layer(x)

class ODEFunc(nn.Module):
    """ ODE function that we need to solve using solver
        Modify from GResBlock
    """
    def __init__(self, in_channel, out_channel, kernel_size=None,
                 padding=1, stride=1, n_class=96, bn=True,
                 activation=F.relu, upsample_factor=2, downsample_factor=1):
        super().__init__()

        self.upsample_factor = upsample_factor if downsample_factor is 1 else 1
        self.downsample_factor = downsample_factor
        self.activation = activation
        self.bn = bn if downsample_factor is 1 else False
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_class = n_class

        self.nfe = 0

        if kernel_size is None:
            kernel_size = [3, 3]
        
        self.conv0 = SpectralNorm(Conv2dODE(out_channel, out_channel, 
                                            kernel_size, stride, padding, 
                                            bias = True))

        self.conv1 = SpectralNorm(Conv2dODE(out_channel, out_channel, 
                                            kernel_size, stride, padding,
                                            bias = True))
        

        if bn:
        #     self.CBNorm1 = ConditionalNorm(in_channel, n_class) # TODO 2 x noise.size[1]
            self.CBNorm = ConditionalNorm(out_channel, n_class)
        
    def forward(self, t, x, condition):
        self.nfe += 1

        out = x
        # print('inside',out.size())
        out = self.conv0(t, out)
        # print(out.size())
        if self.bn:
            out = self.CBNorm(out, condition)

        out = self.activation(out)

        out = self.conv1(t, out)

        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)

        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0,1]).float()

        if self.odefunc.bn:
            self.CBNorm1 = ConditionalNorm(self.odefunc.in_channel, self.odefunc.n_class) # TODO 2 x noise.size[1]
            # self.CBNorm2 = ConditionalNorm(out_channel, n_class)

    def forward(self, x, condition):
        out = x
        if self.odefunc.bn:
            out = self.CBNorm1(out,condition)
    
        out = self.odefunc.activation(out)
        # print(out.size())
        if self.odefunc.upsample_factor != 1:
            out = F.interpolate(out, scale_factor=self.odefunc.upsample_factor)
        print(out.size())
        BT, C, W, H = out.size()
        # zeros augmented
        if self.odefunc.in_channel < self.odefunc.out_channel:
            zeros_aug = torch.zeros([BT, self.odefunc.out_channel - self.odefunc.in_channel, W, H])
            out = torch.cat((out,zeros_aug),1)
        self.integration_time = self.integration_time.type_as(x)
        func = lambda t,x: self.odefunc(t,x,condition)
        out = odeint(func, out, self.integration_time)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


if __name__ == '__main__':
    n_class = 96
    batch_size = 4
    n_frames = 20

    gResBlock = ODEFunc(3, 100, [3, 3])
    odeGResBlock = ODEBlock(gResBlock)
    x = torch.rand([batch_size * n_frames, 3, 64, 64])
    condition = torch.rand([batch_size, n_class])
    condition = condition.repeat(n_frames, 1)
    print(x.size())
    y = odeGResBlock(x,condition)
    print(x.size())
    print(y.size())