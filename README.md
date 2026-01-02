## Tiny Diffusion

This repo contains a minimal text-conditioned diffusion model implementation from scratch. It includes:
- A UNet for latent diffusion
- A VAE to encode/decode images to 4-channel latents
- A small CLIP-style text encoder for conditioning
- Two training/inference schedules: DDPM and Flow Matching


## Directory structure
```bash
diffusion/
├── models/
│   ├── clip.py
│   ├── time_embed.py
│   ├── unet.py
│   └── vae.py
│
├── schedulers/
│   ├── base.py
│   ├── ddpm.py
│   └── flow_matching.py
│
├── .gitignore
├── README.md
├── data.py
├── pyproject.toml
├── requirements.txt
├── sample.py
└── train.py
```

## Requirements
- Python >= 3.12
- PyTorch and torchvision (pinned in `pyproject.toml` / `requirements.txt`)
- A GPU is recommended for training. CPU should be fine for quick tests.
- Datasets are loaded using `datasets` and cached locally (see `--cache_dir`)

## Getting Started

Choose one of the two options below.

### Option A: uv (recommended)
- Install uv: https://docs.astral.sh/uv/
- Run without a manual install, dependencies resolve from `pyproject.toml`:
    ```bash
    uv run train.py --steps 100 --scheduler ddpm
    uv run sample.py --prompt "A small cabin in the woods" --steps 50 --out sample.png
    ```
- **CUDA note:** if you need a specific CUDA wheel, install it once and then run:
  ```bash
  uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
  uv run train.py --steps 100
  ```

### Option B: venv + pip
- Create and activate a virtual environment:
  - 
  ```bash 
  python3 -m venv .venv && source .venv/bin/activate     # macOS/Linux
  python3 -m venv .venv && .venv\\Scripts\\activate      # Windows
  pip install -r requirements.txt                        # install dependencies 

  python3 train.py --steps 100 --scheduler ddpm          # start training 
  python3 sample.py --prompt "A small cabin in the woods" --steps 50 --out sample.png  
  ```

- **CUDA note:** to use a specific CUDA build of PyTorch:
  - `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`

## Training
- Minimal example (DDPM):
  ```bash 
  uv run train.py --steps 100 --scheduler ddpm
  ```
- Flow Matching example:
  ```bash 
  uv run train.py --steps 100 --scheduler flow
  ```
- Common flags (see `train.py`):
  - `--dataset` (default: `flickr30k`)
  - `--split` (default: `train`)
  - `--cache_dir` (default: `./data/hf`)
  - `--image_size` (default: `256`)
  - `--batch_size` (default: `64`)
  - `--subset_size` (0 disables subsetting)
  - `--steps`, `--lr`, `--wd`
  - `--scheduler` (`ddpm` or `flow`)
  - `--prediction_type` (DDPM only: `epsilon`, `x0`, `v_prediction`)

## Sampling
- DDPM:
    ```bash 
    uv run sample.py --prompt "A watercolor landscape" --steps 50 --out out_ddpm.png --scheduler ddpm
    ```
- Flow Matching:
  ```bash 
  uv run sample.py --prompt "A watercolor landscape" --steps 50 --out out_flow.png --scheduler flow
  ```

## Datasets
- Suggested names: `flickr30k`, `flickr8k`, `coco`, `coco_captions`, `vizwiz_caption`
- Images are resized and center-cropped to `--image_size` & pixel range normalized to `[-1, 1]`
- Use `--cache_dir` to control `datasets` cache location.`--subset_size` can be used to iterate quickly.
