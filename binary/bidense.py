import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBias(nn.Module):
    def __init__(self, out_channels):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    

class LearnableBias1D(nn.Module):
    def __init__(self, out_channels):
        super(LearnableBias1D, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class ApproxSignBinarizer(nn.Module):
    def __init__(self):
        super(ApproxSignBinarizer, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        x = out_forward.detach() - out3.detach() + out3

        return x    
    

class ValueAdaptiveBinarizer(nn.Module):
    def __init__(self, dim):
        super(ValueAdaptiveBinarizer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.randn(1, dim, 1, 1) * 1e-3, requires_grad=True)
        self.k = nn.Parameter(torch.randn(1, dim, 1, 1) * 1e-3, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1), requires_grad=True)

        self.binarizer = ApproxSignBinarizer()

    def forward(self, x):
        x = x - (self.k * self.avg_pool(x.detach()) + self.b)
        x = torch.exp(self.alpha * self.avg_pool(x.detach().abs())) * self.binarizer(x)
        return x
    

class ValueAdaptiveBinarizer1D(nn.Module):
    def __init__(self, dim):
        super(ValueAdaptiveBinarizer1D, self).__init__()
        self.alpha = nn.Parameter(torch.randn(dim) * 1e-3, requires_grad=True)
        self.k = nn.Parameter(torch.randn(dim) * 1e-3, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=True)

        self.binarizer = ApproxSignBinarizer()

    def forward(self, x):
        x = x - (self.k * torch.mean(x.detach(), 1, keepdim=True) + self.b)
        x = torch.exp(self.alpha * torch.mean(x.detach().abs(), 1, keepdim=True)) * self.binarizer(x)
        return x


class BinaryWeightConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BinaryWeightConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        nn.init.normal_(self.weight, std=1e-3)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return y


class BinaryWeightConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(BinaryWeightConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )
        nn.init.normal_(self.weight, std=1e-3)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv_transpose2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, 
                               output_padding=self.output_padding, groups=self.groups, dilation=self.dilation)

        return y
    

def channel_adaptive_bypass(x: torch.Tensor, out_ch: int):
    in_ch = x.size(1)
    repeat, merge = out_ch // in_ch, out_ch % in_ch

    out = []
    if repeat > 0:
        out.append(x.repeat(1, repeat, 1, 1))
    
    if merge > 0:
        # append regular merging channels
        regular_ch = (in_ch // merge) * (merge - 1)
        out.append(x[:, :regular_ch].unflatten(1, (-1, merge - 1)).mean(1))
        out.append(x[:, regular_ch:].unflatten(1, (-1, 1)).mean(1))
        
    out = torch.cat(out, 1)
    return out


def channel_adaptive_bypass_1d(x: torch.Tensor, out_ch: int):
    in_ch = x.size(-1)
    repeat, merge = out_ch // in_ch, out_ch % in_ch

    out = []
    if repeat > 0:
        out.append(x.repeat(1, 1, repeat))
    
    if merge > 0:
        # append regular merging channels
        regular_ch = (in_ch // merge) * (merge - 1)
        out.append(x[..., :regular_ch].unflatten(-1, (merge - 1, -1)).mean(-1))
        out.append(x[..., regular_ch:].unflatten(-1, (1, -1)).mean(-1))
        
    out = torch.cat(out, -1)
    return out


class BiDenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, bypass=True):
        super(BiDenseConv2d, self).__init__()
        self.binarizer = ValueAdaptiveBinarizer(in_channels)
        self.conv = BinaryWeightConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.out_channels = out_channels
        self.bypass = bypass
    
    def forward(self, x):
        input = x
        x = self.binarizer(x)
        x = self.conv(x)
        x = self.norm(x)
        if self.bypass:
            x = x + channel_adaptive_bypass(input, self.out_channels)

        return x
    

class BiDenseConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, bypass=True):
        super(BiDenseConvTranspose2d, self).__init__()
        self.binarizer = ValueAdaptiveBinarizer(in_channels)
        self.conv_transpose = BinaryWeightConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.out_channels = out_channels
        self.bypass = bypass
    
    def forward(self, x):
        input = x
        x = self.binarizer(x)
        x = self.conv_transpose(x)
        x = self.norm(x)
        if self.bypass:
            x = x + channel_adaptive_bypass(input, self.out_channels)

        return x
    

class BinaryLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, bypass=True):
        super(BinaryLinear, self).__init__(
            in_channels,
            out_channels,
            bias=bias,
        )
        # self.binarizer = ValueAdaptiveBinarizer1D(in_channels)
        self.move = LearnableBias1D(in_channels)
        self.binarizer = ApproxSignBinarizer()
        if bypass:
            self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.out_channels = out_channels
        self.bypass = bypass

        nn.init.normal_(self.weight, std=1e-3)

    def forward(self, input):
        scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        ba = self.binarizer(self.move(input))
        out = nn.functional.linear(ba, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        if self.bypass:
            out = self.norm(out)
            out = out + channel_adaptive_bypass_1d(input, self.out_channels)
        
        return out
    