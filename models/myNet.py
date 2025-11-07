from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.CGMamba import CGMambaLayer
from einops import rearrange

class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
            input = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

            x = input + self.drop_path(x)
            return x

class CGMambaSeq(nn.Module):
    def __init__(self, dim, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CGMambaLayer(dim, dim) for _ in range(num_blocks)
        ])

    def forward(self, x, H, W):
        for block in self.blocks:
            x = block(x, H, W)
        return x


class CGMambaNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=7,
                 depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., pretrained_cfg=None
                 ):
        super().__init__()

        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stage blocks with learnable alpha
        self.stages = nn.ModuleList()
        self.cgMamba_blocks = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # Define the number of blocks for each stage: 1, 1, 1, 1
        num_blocks_per_stage = [1, 1, 1, 1]
        
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNextBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)

            self.cgMamba_blocks.append(CGMambaSeq(dims[i], num_blocks=num_blocks_per_stage[i]))
            cur += depths[i]

        # Final layers
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # Init
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            
            conv_out = self.stages[i](x)
            
            H, W = x.shape[2], x.shape[3]

            mamba_input = rearrange(x, 'b c h w -> b (h w) c')
            mamba_out = self.cgMamba_blocks[i](mamba_input, H, W)
            mamba_out = rearrange(mamba_out, 'b (h w) c -> b c h w', h=H, w=W)

            x = conv_out + mamba_out

        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
}



@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = CGMambaNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model