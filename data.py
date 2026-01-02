from typing import Iterator, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as T
from PIL import Image

class DatasetCaptions(Dataset):

    def __init__(
        self,
        name: str = "lmms-lab/flickr30k",
        split: str = "train",
        cache_dir: Optional[str] = None,
        image_size: int = 256,
        subset_size: Optional[int] = None,
    ):
        self.PIL = Image
        self.transforms = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.ds = load_dataset(name, split=split, cache_dir=cache_dir)
        if subset_size is not None and subset_size > 0:
            self.ds = self.ds.select(range(min(subset_size, len(self.ds))))

    def __len__(self) -> int:
        return len(self.ds)

    def get_caption(self, example: dict) -> str:
        if "caption" in example and isinstance(example["caption"], str):
            return example["caption"].strip()
        
        if "captions" in example and example["captions"] is not None:
            caps = example["captions"]
            if isinstance(caps, list) and len(caps) > 0:
                first = caps[0]
                if isinstance(first, str):
                    return first.strip()
                if isinstance(first, dict):
                    for key in ("raw", "text", "caption"):
                        if key in first and isinstance(first[key], str):
                            return first[key].strip()

        if "sentence" in example and isinstance(example["sentence"], str):
            return example["sentence"].strip()
        if "sentences" in example and isinstance(example["sentences"], list) and len(example["sentences"]) > 0:
            first = example["sentences"][0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                for key in ("raw", "text", "caption"):
                    if key in first and isinstance(first[key], str):
                        return first[key].strip()

        if "text" in example and isinstance(example["text"], str):
            return example["text"].strip()

        return ""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        ex = self.ds[int(idx)]
        img = ex.get("image")
        if not isinstance(img, self.PIL.Image):
            try:
                img = self.PIL.fromarray(img)
            except Exception:
                pass
        img_t = self.transforms(img)
        caption = self.get_caption(ex)
        return img_t, caption


def get_dataloader(
    name: str = "flickr30k",
    split: str = "train",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 256,
    subset_size: Optional[int] = None,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    ds = DatasetCaptions(name=name, split=split, cache_dir=cache_dir, image_size=image_size, subset_size=subset_size)

    is_mps = torch.backends.mps.is_available()
    use_pin_memory = torch.cuda.is_available() and not is_mps
    effective_workers = 0 if is_mps else max(0, num_workers)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
        persistent_workers=False if effective_workers == 0 else True,
    )
    return loader
