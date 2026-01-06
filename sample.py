
import argparse
from typing import List, Optional

import torch

from models.vae import AutoencoderKL
from models.unet import TinyUNet
from models.clip import Config as CLIPConfig, get_tokenizer_and_encoder
from schedulers.ddpm import DDPMScheduler
from schedulers.flow_matching import FlowMatchingScheduler
from PIL import Image

def save_image(x: torch.Tensor, path: str) -> None:

    x = x.clamp(0, 1)
    x = (x * 255.0).byte().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.permute(1, 2, 0)  # HWC
    img = Image.fromarray(x.numpy())
    img.save(path)


@torch.no_grad()
def sample(prompt: str, 
        steps: int = 50, 
        seed: Optional[int] = 0, 
        out: Optional[str] = None, 
        scheduler_type: str = "ddpm",
        vocab_path: str = None,
        merges_path: str = None) -> torch.Tensor:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    # Text encoder + tokenizer
    clip_cfg = CLIPConfig(
        context_length=77,
        vocab_json_path=vocab_path,
        merges_txt_path=merges_path,
        pad_token=None,
        bos_token=None,
        eos_token="</s>",
        model_width=512,
        layers=12,
        heads=8,
    )
    text_enc, tokenizer = get_tokenizer_and_encoder(clip_cfg)
    text_enc.to(device).eval()

    # VAE
    vae = AutoencoderKL(
        img_ch=3,
        latent_ch=4,
        base_ch=128,
        ch_mult=(1, 2, 2),  # 256 -> 64
        attn_res=(16,),
        latent_scaling=0.18215,
    ).to(device).eval()

    # UNet
    text_ctx_dim = getattr(text_enc, "width", 512)
    unet = TinyUNet(
        in_channels=4,
        out_channels=4,
        base_channels=320,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        time_emb_dim=1280,
        text_ctx_dim=text_ctx_dim,
        attn_resolutions=(16, 8),
    ).to(device).eval()

    # Scheduler
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(num_train_timesteps=1000, device=device)
        scheduler.set_timesteps(steps)
    elif scheduler_type == "flow":
        scheduler = FlowMatchingScheduler(timesteps=1000, device=device)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    # Condition
    token_ids = tokenizer([prompt]).to(device)
    token_feats, _ = text_enc(token_ids)  # (1, L, D)

    B, H, W = 1, 64, 64
    x = torch.randn(B, 4, H, W, device=device)

    if scheduler_type == "ddpm":
        for t in scheduler.timesteps:
            t_batch = t.repeat(B)
            eps = unet(x, t_batch, token_feats)
            x = scheduler.step(eps, t_batch, x)
    else:
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
        for i in range(steps):
            t_batch = ts[i].expand(B)
            dt = (ts[i] - ts[i + 1]).abs()
            v = unet(x, t_batch, token_feats)
            x = x - dt * v

    # Decode to image
    img = vae.decode(x)
    img = (img + 1) / 2 

    if out:
        save_image(img, out)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="A beautiful landscape painting")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="sample.png")
    p.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "flow"])
    p.add_argument("--vocab-path", type=str, default=None)
    p.add_argument("--merges-path", type=str, default=None)

    args = p.parse_args()

    img = sample(args.prompt, steps=args.steps, seed=args.seed, out=args.out, scheduler_type=args.scheduler, vocab_path=args.vocab_path, merges_path=args.merges_path)
    print(f"[{args.scheduler}] Generated image tensor: {tuple(img.shape)} saved to {args.out}")


if __name__ == "__main__":
    main()
