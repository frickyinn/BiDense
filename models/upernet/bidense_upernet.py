import torch
import torch.nn as nn
import torch.nn.functional as F

from binary.cfb import CFBConv2d
from binary.bidense import BiDenseConv2d, LearnableBias
from .backbones.bidense_convnext import get_convnext


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, out_channel=None, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            CFBConv2d(in_channels + (out_channels * len(bin_sizes)), out_channel if out_channel else in_channels, kernel_size=3, padding=1, bias=False),
            LearnableBias(out_channel if out_channel else in_channels),
            nn.PReLU(out_channel if out_channel else in_channels),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = CFBConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        move = LearnableBias(out_channels)
        prelu = nn.PReLU(out_channels)
        return nn.Sequential(prior, conv, move, prelu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output
     

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([CFBConv2d(ft_size, fpn_out, kernel_size=1) for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([CFBConv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            CFBConv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            LearnableBias(fpn_out),
            nn.PReLU(fpn_out),
            LearnableBias(fpn_out),
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, head, in_channels=3, features=256, backbone='convnext_tiny', pretrained=False):
        super(UperNet, self).__init__()

        if 'convnext' in backbone:
            self.backbone = get_convnext(in_chans=in_channels, model_name=backbone, pretrained=pretrained)
            feature_channels = self.backbone.dims

        fpn_out = feature_channels[0]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)

        self.out_conv1 = nn.Conv2d(fpn_out, features // 2, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = head

    def forward(self, x):
        B, _, H, W = x.shape
        # input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        
        x = self.FPN(features)
        x = self.out_conv1(x)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=True)
        x = self.out_conv2(x)

        return x


class BiDenseUperNetDepthModel(UperNet):
    def __init__(self, max_depth, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256
        self.max_depth = max_depth

        head = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        super().__init__(head, **kwargs)

    def forward(self, x):
        depth = super().forward(x) * self.max_depth

        return depth


class BiDenseUperNetSegmentationModel(UperNet):
    def __init__(self, num_classes, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        head = nn.Sequential(
            nn.Conv2d(features // 2, features // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, num_classes, kernel_size=1),
        )

        super().__init__(head, **kwargs)
