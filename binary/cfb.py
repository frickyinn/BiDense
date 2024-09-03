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


class ReActConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ReActConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.move = LearnableBias(in_channels)
        nn.init.normal_(self.weight, std=1e-3)

    def forward(self, x):
        x = self.move(x)

        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        x = out_forward.detach() - out3.detach() + out3

        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return y
    

def resize_channels(x: torch.Tensor, out_ch: int):
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


class CFBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False, bypass=True):
        super(CFBConv2d, self).__init__()
        self.conv = ReActConv2d(
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
        x = self.conv(x)
        x = self.norm(x)
        if self.bypass:
            x = x + resize_channels(input, self.out_channels)

        return x
    