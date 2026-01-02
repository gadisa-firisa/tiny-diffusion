import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_embed import SiLU, TimeEmbedding


class TextProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.proj(context)


class FiLMAdapter(nn.Module):
    def __init__(self, in_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(in_dim, channels)
        self.to_beta = nn.Linear(in_dim, channels)

    def forward(self, text_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = self.to_gamma(text_global)
        beta = self.to_beta(text_global)
        return gamma, beta


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# Unet residual block 
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_film: bool = False,
        film_in_dim: Optional[int] = None,
        groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_film = use_film

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.silu = SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        if use_film:
            assert film_in_dim is not None
            self.film = FiLMAdapter(film_in_dim, out_channels)
        else:
            self.film = None
    
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_global: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, C, H, W)
        h = self.norm1(x)
        h = self.silu(h)
        h = self.conv1(h)

        # time
        t = self.time_proj(t_emb)  # (B, out_channels)
        h = h + t[:, :, None, None]

        if self.use_film and text_global is not None:
            gamma, beta = self.film(text_global)   # (B, C), (B, C)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        h = self.norm2(h)
        h = self.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape to (B, heads, HW, dim)
        def reshape(t):
            t = t.reshape(B, self.num_heads, self.head_dim, H * W)
            return t.permute(0, 1, 3, 2)  # (B, heads, HW, dim)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, heads, HW, HW)

        attn = attn_scores.softmax(dim=-1)
        out = attn @ v  # (B, heads, HW, dim)

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out  # residual


class CrossAttention(nn.Module):
    
    def __init__(self, dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, D)
        context: torch.Tensor,     # (B, L, D_ctx)
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        q = self.to_q(x).reshape(B, N, H, self.head_dim).transpose(1, 2)   # (B, H, N, d)
        k = self.to_k(context).reshape(B, -1, H, self.head_dim).transpose(1, 2)  # (B, H, L, d)
        v = self.to_v(context).reshape(B, -1, H, self.head_dim).transpose(1, 2)  # (B, H, L, d)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, H, N, L)

        attn = attn_scores.softmax(dim=-1)
        out = attn @ v  # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out + x  # residual


class AttnBlock2d(nn.Module):

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.cross_attn = CrossAttention(channels, context_dim, num_heads)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = self.to_q(h)   # (B, C, H, W)
        h_flat = h.flatten(2).transpose(1, 2)  # (B, HW, C)
        h_attn = self.cross_attn(h_flat, context)  # (B, HW, C)
        h_attn = h_attn.transpose(1, 2).reshape(B, C, H, W)
        h_attn = self.proj_out(h_attn)
        return x + h_attn


class TinyUNet(nn.Module):
   
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 320,
        channel_mults=(1, 2, 4, 4),   # 64,32,16,8
        num_res_blocks: int = 2,
        time_emb_dim: int = 1280,
        text_ctx_dim: int = 768,      # frozen encoder output dim
        attn_resolutions=(16, 8),
        self_attn_resolutions: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_resolutions = attn_resolutions
        self.self_attn_resolutions = (
            attn_resolutions if self_attn_resolutions is None else self_attn_resolutions
        )

        # time embedding 
        self.time_embed = TimeEmbedding(
            output_dim=time_emb_dim,
            embed_dim=base_channels,
            hidden_dim=time_emb_dim,
        )

        # text projection for cross-attn
        self.text_proj = TextProj(text_ctx_dim, base_channels * 4)  
        self.text_global_proj = nn.Linear(text_ctx_dim, base_channels * 4)

        # input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)


        # DOWN
        self.down = nn.ModuleList()

        ch = base_channels
        ds = 1  # downsample factor relative to 64

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                use_film = (i == 0)
                blocks.append(
                    ResBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        time_emb_dim=time_emb_dim,
                        use_film=use_film,
                        film_in_dim=self.text_global_proj.out_features if use_film else None,
                    )
                )
                ch = out_ch

            res = 64 // ds
            attn = None
            self_attn = None
            if res in attn_resolutions:
                attn = AttnBlock2d(ch, context_dim=self.text_proj.proj.out_features)
            if res in self.self_attn_resolutions:
                self_attn = SelfAttention(ch)

            downsample = None
            if i != len(channel_mults) - 1:
                downsample = Downsample(ch)
                ds *= 2

            self.down.append(
                nn.ModuleDict(
                    dict(
                        blocks=blocks,
                        attn=attn,
                        self_attn=self_attn,
                        downsample=downsample,
                    )
                )
            )


        # MIDDLE
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim=time_emb_dim)
        mid_res = 64 // (2 ** (len(channel_mults) - 1))
        self.mid_self_attn = SelfAttention(ch) if mid_res in self.self_attn_resolutions else None
        self.mid_attn = AttnBlock2d(ch, context_dim=self.text_proj.proj.out_features)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim=time_emb_dim)


        # UP
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        in_channels=ch + out_ch,
                        out_channels=out_ch,
                        time_emb_dim=time_emb_dim,
                    )
                )
                ch = out_ch

            res = 64 // (2 ** i)
            attn = None
            self_attn = None
            if res in attn_resolutions:
                attn = AttnBlock2d(ch, context_dim=self.text_proj.proj.out_features)
            if res in self.self_attn_resolutions:
                self_attn = SelfAttention(ch)

            upsample = None
            if i != 0:
                upsample = Upsample(ch)

            self.up.append(
                nn.ModuleDict(
                    dict(
                        blocks=blocks,
                        attn=attn,
                        self_attn=self_attn,
                        upsample=upsample,
                    )
                )
            )

        # output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,              # (B, 4, 64, 64)
        t: torch.Tensor,              # (B,)
        context: torch.Tensor,        # (B, L, D_txt)
    ) -> torch.Tensor:

        t_emb = self.time_embed(t)  # (B, time_emb_dim)
        ctx = self.text_proj(context)  # (B, L, D_ctx_proj)
        text_global = context.mean(dim=1)
        text_global = self.text_global_proj(text_global)  # (B, Dg)

        # input
        h = self.input_conv(x)
        hs = []

        # down
        for stage in self.down:
            for block in stage["blocks"]:
                h = block(h, t_emb, text_global)
                hs.append(h)
            if stage.get("self_attn") is not None:
                h = stage["self_attn"](h)
            if stage["attn"] is not None:
                h = stage["attn"](h, ctx)
            if stage["downsample"] is not None:
                h = stage["downsample"](h)

        # middle
        h = self.mid_block1(h, t_emb, text_global)
        if self.mid_self_attn is not None:
            h = self.mid_self_attn(h)
        h = self.mid_attn(h, ctx)
        h = self.mid_block2(h, t_emb, text_global)

        # up 
        for stage in self.up:
            for block in stage["blocks"]:
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb, text_global)
            if stage.get("self_attn") is not None:
                h = stage["self_attn"](h)
            if stage["attn"] is not None:
                h = stage["attn"](h, ctx)
            if stage["upsample"] is not None:
                h = stage["upsample"](h)

        # output
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h

    
