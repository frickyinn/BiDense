import torch
import torch.nn as nn
import torch.nn.functional as F

import thop


class HardBinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(HardBinaryConv2d, self).__init__(
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
        binary_input_no_grad = torch.sign(x)
        cliped_input = torch.clamp(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return y
    

def count_bnn_convNd(m, x, y: torch.Tensor):
        x = x[0]

        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        m.total_ops += thop.vision.calc_func.calculate_conv2d_flops(
            input_size = list(x.shape),
            output_size = list(y.shape),
            kernel_size = list(m.weight.shape),
            groups = m.groups,
            bias = m.bias
        ) / 64


class HardBinaryLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super(HardBinaryLinear, self).__init__(
            in_channels,
            out_channels,
            bias=bias,
        )
        nn.init.normal_(self.weight, std=1e-3)

    def forward(self, x):
        binary_input_no_grad = torch.sign(x)
        cliped_input = torch.clamp(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight
        binary_weights_no_grad = torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.linear(x, binary_weights, self.bias)

        return y