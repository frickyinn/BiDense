import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from binary.adabin import AdaBinConv2d, Maxout


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = AdaBinConv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = AdaBinConv2d(dim, 4 * dim, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(4 * dim)
        self.act = Maxout(4 * dim)
        self.pwconv2 = AdaBinConv2d(4 * dim, dim, kernel_size=1, stride=1)
        self.norm3 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = x + self.norm1(self.dwconv(x))
        x = self.pwconv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.norm3(x)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, to_cls=False,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()
        self.dims = dims
        self.to_cls = to_cls

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                AdaBinConv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                nn.BatchNorm2d(dims[i+1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        _, C, H, W = x.shape
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        if not self.to_cls:
            return features
        else:
            x = self.head(self.norm(x.mean([-2,-1])))
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def convnext_small(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_base(pretrained=False, in_22k=False,to_cls=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], to_cls=to_cls, **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def get_convnext(model_name='convnext_tiny', pretrained=True, in_22k=True, in_chans=3, scale=4, to_cls=False, **kwargs):
    if model_name == 'convnext_tiny':
        model = convnext_tiny(pretrained=pretrained, **kwargs)
    elif model_name == 'convnext_small':
        model = convnext_small(pretrained=pretrained, **kwargs)
    elif model_name == 'convnext_base_1k':
        model = convnext_base(pretrained=pretrained, in_22k=False, **kwargs)
    elif model_name == 'convnext_base_22k':
        model = convnext_base(pretrained=pretrained, in_22k=True, num_classes=21841, to_cls=to_cls, **kwargs)
    elif model_name == 'convnext_large_1k':
        model = convnext_large(pretrained=pretrained, in_22k=False, **kwargs)
    elif model_name == 'convnext_large_22k':
        model = convnext_large(pretrained=pretrained, in_22k=True, num_classes=21841, **kwargs)
    elif model_name == 'convnext_xlarge_22k':
        model = convnext_xlarge(pretrained=pretrained, in_22k=True, num_classes=21841, **kwargs)
    else:
        raise NotImplementedError(f"Unkown model: {model_name}")
    if in_chans != 3:
        stem = nn.Sequential(
            nn.Conv2d(in_chans, model.dims[0], kernel_size=scale, stride=scale),
            nn.BatchNorm2d(model.dims[0])
        )
        model.downsample_layers[0] = stem
    if to_cls:
        model.head = nn.Linear(model.dims[-1], 1)

    return model
