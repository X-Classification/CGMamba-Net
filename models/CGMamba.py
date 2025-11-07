import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    from .ss2d import SS2D
    from .csms6s import (
        CrossScan_Concentric_Clockwise, CrossScan_Convergence_Clockwise,
        CrossScan_Concentric_CounterClockwise, CrossScan_Convergence_CounterClockwise,
        CrossMerge_Concentric_Clockwise, CrossMerge_Convergence_Clockwise,
        CrossMerge_Concentric_CounterClockwise, CrossMerge_Convergence_CounterClockwise
    )
except:
    from ss2d import SS2D
    from csms6s import (
        CrossScan_Concentric_Clockwise, CrossScan_Convergence_Clockwise,
        CrossScan_Concentric_CounterClockwise, CrossScan_Convergence_CounterClockwise,
        CrossMerge_Concentric_Clockwise, CrossMerge_Convergence_Clockwise,
        CrossMerge_Concentric_CounterClockwise, CrossMerge_Convergence_CounterClockwise
    )

from einops import rearrange

class CGMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()

        num_channels_reduced = input_dim // reduction
        self.fc1 = nn.Linear(input_dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, output_dim, bias=True)
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba_g1 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g2 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g3 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g4 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, H, W):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, N, C = x.shape
        indentity = x
        x = self.norm(x)

        # Channel Affinity
        z = x.permute(0, 2, 1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        x = rearrange(x, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)

        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_Concentric_Clockwise, CrossMerge=CrossMerge_Concentric_Clockwise)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_Convergence_Clockwise, CrossMerge=CrossMerge_Convergence_Clockwise)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_Concentric_CounterClockwise, CrossMerge=CrossMerge_Concentric_CounterClockwise)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_Convergence_CounterClockwise, CrossMerge=CrossMerge_Convergence_CounterClockwise)


        # Combine all feature maps
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1) * self.skip_scale * x

        x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)

        # Channel Modulation
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        
        return self.alpha * x_mamba + indentity