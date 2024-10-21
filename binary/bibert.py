import torch
import torch.nn as nn
from torch.nn import Parameter


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class BinaryLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True):
        super(BinaryLinear, self).__init__(
            in_channels,
            out_channels,
            bias=bias,
        )

    def forward(self, input):
        scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        binary_input_no_grad = torch.sign(input)
        cliped_input = torch.clamp(input, -1.0, 1.0)
        ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        
        out = nn.functional.linear(ba, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out
    