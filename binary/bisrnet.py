import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class ReDistribution(nn.Module):
    def __init__(self, out_chn):
        super(ReDistribution, self).__init__()
        self.b = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)
        self.k = nn.Parameter(torch.ones(1, out_chn, 1, 1), requires_grad=True)
    
    def forward(self, x):
        out = x * self.k.expand_as(x) + self.b.expand_as(x)
        return out
    

class Spectral_Binary_Activation(nn.Module):
    def __init__(self):
        super(Spectral_Binary_Activation, self).__init__()
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        binary_activation_no_grad = torch.sign(x)
        tanh_activation = torch.tanh(x*self.beta)
        out = binary_activation_no_grad.detach() - tanh_activation.detach() + tanh_activation

        return out
    

class BiSRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BiSRConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.move = ReDistribution(in_channels)
        self.binary_activation = Spectral_Binary_Activation()
        nn.init.normal_(self.weight, std=1e-3)

    def forward(self, x):
        x = self.move(x)
        x = self.binary_activation(x)

        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return y
    

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x


class BiSRConv2d_Down(nn.Module):
    '''
    Binarized DownSample from BiSRNet
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3 , stride=1, padding=1, bias=False, groups=1):
        assert out_channels / in_channels == 2
        super(BiSRConv2d_Down, self).__init__()

        self.biconv_1 = BiSRConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.biconv_2 = BiSRConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self.rprelu1 = RPReLU(in_channels)
        self.rprelu2 = RPReLU(in_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)
        out_1 = out
        out_2 = out_1.clone()
        out_1 = out_1 + self.rprelu1(self.biconv_1(out_1))
        out_2 = out_2 + self.rprelu2(self.biconv_2(out_2))
        out = torch.cat([out_1, out_2], dim=1)

        return out
    