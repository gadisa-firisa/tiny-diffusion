import argparse
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from models.vae import AutoencoderKL
from models.unet import TinyUNet
from models.clip import build_text_stack
from schedulers.ddpm import DDPMScheduler
from schedulers.flow_matching import FlowMatchingScheduler
from data import get_dataloader

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--dataset", type=str, default="flickr30k")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--cache_dir", type=str, default="./data/hf")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--subset_size", type=int, default=0, help="0 disables subsetting")
    # training
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    # scheduler/model
    p.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "flow"])
    p.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "x0", "v_prediction"], help="For DDPM-like schedulers")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    text_enc, tokenizer = build_text_stack(
        vocab_json_path= "",
        merges_txt_path= "",
        context_length=77,
    )
    text_enc.to(device)
    text_enc.eval()  

    vae = AutoencoderKL(
        img_ch=3,
        latent_ch=4,
        base_ch=128,
        ch_mult=(1, 2, 2),  # 256 -> 128 -> 64
        attn_res=(16,),
        latent_scaling=0.18215,
    ).to(device)

    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()


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
    ).to(device)

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.scheduler == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="linear",
            prediction_type=args.prediction_type,
            device=device,
        )
    else:
        scheduler = FlowMatchingScheduler(timesteps=1000, device=device)

    batch_size = args.batch_size
    image_size = args.image_size
    subset_size = args.subset_size if args.subset_size > 0 else None
    loader = iter(
        get_dataloader(
            name=args.dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            batch_size=batch_size,
            image_size=image_size,
            subset_size=subset_size,
        )
    )

    # Training loop
    unet.train()
    num_steps = args.steps
    for step in range(num_steps):
        try:
            images, prompts = next(loader)
        except StopIteration:
            loader = iter(get_dataloader(
                name=args.dataset,
                split=args.split,
                cache_dir=args.cache_dir,
                batch_size=batch_size,
                image_size=image_size,
                subset_size=subset_size,
            ))
            images, prompts = next(loader)
        images = images.to(device)

        # Encode text (prompts)
        with torch.no_grad():
            token_ids = tokenizer(prompts).to(device)
            token_feats, _ = text_enc(token_ids)  # (B, L, D), (B, D)

        # Encode images
        with torch.no_grad():
            z_scaled, mu, logvar = vae.encode(images)

        B = z_scaled.size(0)
        noise = torch.randn_like(z_scaled)

        if args.scheduler == "ddpm":
            t = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device, dtype=torch.long)
            z_noisy = scheduler.add_noise(z_scaled, noise, t)
            eps_pred = unet(z_noisy, t, token_feats)
            loss = nn.functional.mse_loss(eps_pred, noise)
        else:  # flow matching
            t = torch.rand(B, device=device)
            x_t = scheduler.forward(z_scaled, t, noise)
            v_target = noise - z_scaled
            v_pred = unet(x_t, t, token_feats)
            loss = nn.functional.mse_loss(v_pred, v_target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"step {step+1}/{num_steps} | {args.scheduler} | loss {loss.item():.4f}")



if __name__ == "__main__":
    main()
