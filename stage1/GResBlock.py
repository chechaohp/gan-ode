# original from https://github.com/Harrypotterrrr/DVD-GAN
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ConditionalNorm(nn.Module):

    def __init__(self, in_channel, n_condition=96):
        super().__init__()

        self.in_channel = in_channel
        self.bn = nn.BatchNorm2d(self.in_channel, affine=False)

        self.embed = nn.Linear(n_condition, self.in_channel * 2)
        self.embed.weight.data[:, :self.in_channel].normal_(1, 0.02)
        self.embed.weight.data[:, self.in_channel:].zero_()

    def forward(self, x, class_id):
        out = self.bn(x)
        embed = self.embed(class_id)
        gamma, beta = embed.chunk(2, 1)
        # gamma = gamma.unsqueeze(2).unsqueeze(3)
        # beta = beta.unsqueeze(2).unsqueeze(3)
        gamma = gamma.view(-1, self.in_channel, 1, 1)
        beta = beta.view(-1, self.in_channel, 1, 1)
        out = gamma * out + beta

        return out

class GResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=None,
                 padding=1, stride=1, n_class=96, bn=True,
                 activation=F.relu, upsample_factor=2, downsample_factor=1):
        super().__init__()

        self.upsample_factor = upsample_factor if downsample_factor is 1 else 1
        self.downsample_factor = downsample_factor
        self.activation = activation
        self.bn = bn if downsample_factor is 1 else False

        if kernel_size is None:
            kernel_size = [3, 3]

        self.conv0 = SpectralNorm(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))
        self.conv1 = SpectralNorm(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=True if bn else True))

        self.skip_proj = True
        self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))

        # if in_channel != out_channel or upsample_factor or downsample_factor:
        #     self.conv_sc = SpectralNorm(nn.Conv2d(in_channel, out_channel, 1, 1, 0))
        #     self.skip_proj = True

        if bn:
            self.CBNorm1 = ConditionalNorm(in_channel, n_class) # TODO 2 x noise.size[1]
            self.CBNorm2 = ConditionalNorm(out_channel, n_class)

    def forward(self, x, condition=None):

        # The time dimension is combined with the batch dimension here, so each frame proceeds
        # through the blocks independently
        BT, C, W, H = x.size()
        out = x

        if self.bn:
            out = self.CBNorm1(out, condition)

        out = self.activation(out)

        if self.upsample_factor != 1:
            out = F.interpolate(out, scale_factor=self.upsample_factor)

        out = self.conv0(out)

        if self.bn:
            out = out.view(BT, -1, W * self.upsample_factor, H * self.upsample_factor)
            out = self.CBNorm2(out, condition)

        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample_factor != 1:
            out = F.avg_pool2d(out, self.downsample_factor)

        if self.skip_proj:
            skip = x
            if self.upsample_factor != 1:
                skip = F.interpolate(skip, scale_factor=self.upsample_factor)
            skip = self.conv_sc(skip)
            if self.downsample_factor != 1:
                skip = F.avg_pool2d(skip, self.downsample_factor)
        else:
            skip = x

        y = out + skip
        y = y.view(
            BT, -1,
            W * self.upsample_factor // self.downsample_factor,
            H * self.upsample_factor // self.downsample_factor
        )

        return y


if __name__ == "__main__":

    n_class = 96
    batch_size = 4
    n_frames = 20

    gResBlock = GResBlock(3, 100, [3, 3])
    x = torch.rand([batch_size * n_frames, 3, 64, 64])
    condition = torch.rand([batch_size, n_class])
    condition = condition.repeat(n_frames, 1)
    y = gResBlock(x, condition)
    print(gResBlock)
    print(x.size())
    print(y.size())

    # with SummaryWriter(comment='gResBlock') as w:
    #     w.add_graph(gResBlock, [x, condition, ])