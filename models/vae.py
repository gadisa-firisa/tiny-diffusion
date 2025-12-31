import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from time_embed import SiLU

class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape[:2]
        if c != self.num_channels:
            raise ValueError("input channel count does not match num_channels")
        orig_shape = x.shape
        x = x.reshape(n, self.num_groups, -1)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        x = x.reshape(orig_shape)
        if self.weight is not None:
            shape = [1, c] + [1] * (x.ndim - 2)
            x = x * self.weight.view(*shape) + self.bias.view(*shape)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=32):
        super().__init__()
        self.norm1 = GroupNorm(groups, in_ch)
        self.silu = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        h = self.norm1(x)
        h = self.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownsampleVAE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpsampleVAE(nn.Module):
    def __init__(self, ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    
    def __init__(self, ch, num_heads=4):
        super().__init__()
        assert ch % num_heads == 0
        self.norm = GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape # B: batch, C: channels, H: height, W: width
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        def reshape(t):
            t = t.reshape(B, self.num_heads, self.head_dim, H * W)
            return t.permute(0, 1, 3, 2)  # (B, heads, HW, dim)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, heads, HW, dim)

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


class EncoderBlock(nn.Module):
    
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 128,
        ch_mult=(1, 2, 4, 4),  # 256->128->64->32->16
        z_channels: int = 4,
        attn_res=(16,),
    ):
        super().__init__()

        # (N, in_ch, 256, 256) -> (N, base_ch, 256, 256)
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        in_chs = [base_ch]
        chs = []
        ds = 1
        self.down = nn.ModuleList()
        cur_ch = base_ch

        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            block1 = ResidualBlock(cur_ch, out_ch)
            block2 = ResidualBlock(out_ch, out_ch)
            blocks = nn.ModuleList([block1, block2])

            attn = None
            res = 256 // ds
            if res in attn_res:
                attn = AttentionBlock(out_ch)

            downsample = None
            if i != len(ch_mult) - 1:
                downsample = DownsampleVAE(out_ch)
                ds *= 2

            self.down.append(
                nn.ModuleDict(
                    dict(
                        blocks=blocks,
                        attn=attn,
                        downsample=downsample,
                    )
                )
            )
            cur_ch = out_ch
            chs.append(cur_ch)
            in_chs.append(cur_ch)


        # (N, cur_ch, 16, 16) -> (N, cur_ch, 16, 16)
        self.mid_block1 = ResidualBlock(cur_ch, cur_ch)

        self.mid_attn = AttentionBlock(cur_ch)
        self.mid_block2 = ResidualBlock(cur_ch, cur_ch)

        self.norm_out = GroupNorm(32, cur_ch)
        self.conv_mu = nn.Conv2d(cur_ch, z_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(cur_ch, z_channels, 3, padding=1)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        for stage in self.down:
            for block in stage["blocks"]:
                h = block(h)
            if stage["attn"] is not None:
                h = stage["attn"](h)
            if stage["downsample"] is not None:
                h = stage["downsample"](h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.norm_out(h)
        h = self.silu(h)

        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class DecoderBlock(nn.Module):

    def __init__(
        self,
        out_ch: int = 3,
        base_ch: int = 128,
        ch_mult=(1, 2, 4, 4),
        z_channels: int = 4,
        attn_res=(16,),
    ):
        super().__init__()

        last_ch = base_ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, last_ch, 3, padding=1)

        self.mid_block1 = ResidualBlock(last_ch, last_ch)
        self.mid_attn = AttentionBlock(last_ch)
        self.mid_block2 = ResidualBlock(last_ch, last_ch)

        self.up = nn.ModuleList()
        ds = 256 // 2 ** (len(ch_mult) - 1)  
        cur_ch = last_ch

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            blocks = nn.ModuleList([
                ResidualBlock(cur_ch, out_ch),
                ResidualBlock(out_ch, out_ch),
            ])

            attn = None
            res = ds
            if res in attn_res:
                attn = AttentionBlock(out_ch)

            upsample = None
            if i != 0:
                upsample = UpsampleVAE(out_ch)
                ds *= 2

            self.up.append(
                nn.ModuleDict(
                    dict(
                        blocks=blocks,
                        attn=attn,
                        upsample=upsample
                    )
                )
            )
            cur_ch = out_ch

        self.norm_out = GroupNorm(32, cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, out_ch, 3, padding=1)
        self.silu = SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        # z *= 1/0.18215  # scale for diffusion
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        ds = 256 // 2 ** (len(self.up)) 
        for stage in self.up:
            for block in stage["blocks"]:
                h = block(h)
            if stage["attn"] is not None:
                h = stage["attn"](h)
            if stage["upsample"] is not None:
                h = stage["upsample"](h)

        h = self.norm_out(h)
        h = self.silu(h)
        x = self.conv_out(h)
        return x


class AutoencoderKL(nn.Module):
   
    def __init__(
        self,
        img_ch: int = 3,
        latent_ch: int = 4,
        base_ch: int = 128,
        ch_mult=(1, 2, 4, 4),
        attn_res=(16,),
        latent_scaling: float = 0.18215,  
    ):
        super().__init__()
        self.encoder = EncoderBlock(
            in_ch=img_ch,
            base_ch=base_ch,
            ch_mult=ch_mult,
            z_channels=latent_ch,
            attn_res=attn_res,
        )
        self.decoder = DecoderBlock(
            out_ch=img_ch,
            base_ch=base_ch,
            ch_mult=ch_mult,
            z_channels=latent_ch,
            attn_res=attn_res,
        )
        self.latent_scaling = latent_scaling

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # scale for diffusion
        z_scaled = z * self.latent_scaling
        return z_scaled, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # unscale
        z = z / self.latent_scaling
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       
        z_scaled, mu, logvar = self.encode(x)
        x_recon = self.decode(z_scaled)
        return x_recon, z_scaled, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL(N(mu, sigma), N(0,1))
        return 0.5 * torch.sum(
            torch.exp(logvar) + mu ** 2 - 1.0 - logvar,
            dim=(1, 2, 3)
        ).mean()



def vae_loss_fn(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
):
    recon_loss = F.l1_loss(x_recon, x)
    kl = AutoencoderKL.kl_loss(mu, logvar)
    loss = recon_loss + kl_weight * kl
    return loss, recon_loss, kl

    
