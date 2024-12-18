import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

from .backbones.bivit_vit import dinov2_vits14
from binary.bidense import BiDenseConv2d, BiDenseConvTranspose2d, BinaryLinear, LearnableBias, channel_adaptive_bypass


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = BiDenseConv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = BiDenseConv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = BiDenseConv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = BiDenseConv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = BiDenseConv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = BiDenseConv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)

        self.prelu1 = nn.PReLU(features)
        self.prelu2 = nn.PReLU(features)

        self.move1 = LearnableBias(features)
        self.move2 = LearnableBias(features)

    def forward(self, x):
        out = self.prelu1(self.move1(x))
        out = self.conv1(out)
        out = self.prelu2(self.move2(out))
        out = self.conv2(out)

        return out


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output
    

class BiDenseUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(BiDenseUp, self).__init__()
        self.conv = BiDenseConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale, padding=0, bypass=False)
        self.out_channels = out_channels
        self.scale = scale

    def forward(self, x):
        return self.conv(x) + channel_adaptive_bypass(F.interpolate(x, scale_factor=self.scale), self.out_channels)
    

class BiDenseDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiDenseDown, self).__init__()
        self.conv = BiDenseConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bypass=False)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        return self.conv(x) + channel_adaptive_bypass(self.avgpool(x), self.out_channels)
    

class DPT(nn.Module):
    def __init__(self, head, features=256, out_channels=[48, 96, 192, 384], use_clstoken=False):
        super(DPT, self).__init__()

        self.use_clstoken = use_clstoken

        self.backbone = dinov2_vits14(pretrained=False)
        in_channels = self.backbone.blocks[0].attn.qkv.in_features

        self.projects = nn.ModuleList([
            BiDenseConv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0) 
            for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            BiDenseUp(in_channels=out_channels[0], out_channels=out_channels[0], scale=4),
            BiDenseUp(in_channels=out_channels[1], out_channels=out_channels[1], scale=2),
            nn.Identity(),
            BiDenseDown(in_channels=out_channels[3], out_channels=out_channels[3])
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        BinaryLinear(2 * in_channels, in_channels),
                        LearnableBias(in_channels),
                        nn.PReLU(in_channels)))
        
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features)

        self.scratch.output_conv1 = BiDenseConv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = head

    def forward(self, x):
        h, w = x.shape[-2:]
        patch_h, patch_w = (h // 28) * 2, (w // 28) * 2

        x = resize(x, (patch_h * 14, patch_w * 14))
        out_features = self.backbone.get_intermediate_layers(x, 4, return_class_token=True)

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)

        return out


class BiDenseDPTDepthModel(DPT):
    def __init__(self, max_depth, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        self.max_depth = max_depth

        head = nn.Sequential(
            BiDenseConv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        super().__init__(head, **kwargs)

    def forward(self, x):
        depth = super().forward(x) * self.max_depth

        return depth


class BiDenseDPTSegmentationModel(DPT):
    def __init__(self, num_classes, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        # kwargs["use_bn"] = True

        head = nn.Sequential(
            BiDenseConv2d(features // 2, features // 2, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, num_classes, kernel_size=1),
        )

        super().__init__(head, **kwargs)
        