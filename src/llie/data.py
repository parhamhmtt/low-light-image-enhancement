"""Paired dark/bright image dataset."""
from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import ToTensor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

class PairedLowLightDataset(Dataset):
    """Expects root/{train,val}/{dark,bright}/ with matching filenames."""
    def __init__(self, root: str | Path, split: str = "train", patch_size: int = 256, augment: bool | None = None):
        self.root = Path(root)
        self.dark_dir = self.root / split / "dark"
        self.bright_dir = self.root / split / "bright"
        self.patch_size = patch_size
        self.augment = split == "train" if augment is None else augment
        if not self.dark_dir.is_dir() or not self.bright_dir.is_dir():
            raise FileNotFoundError(f"Expected paired folders: {self.dark_dir} and {self.bright_dir}")
        self.files = sorted(p.name for p in self.dark_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                            and (self.bright_dir / p.name).is_file())
        if not self.files:
            raise FileNotFoundError(f"No matching dark/bright image pairs found in {self.dark_dir} and {self.bright_dir}")
        self.to_tensor = ToTensor()

    def __len__(self): return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.files[index]
        dark = Image.open(self.dark_dir / name).convert("RGB")
        bright = Image.open(self.bright_dir / name).convert("RGB")
        if dark.size != bright.size:
            raise ValueError(f"Paired images must have identical dimensions: {name}, {dark.size} != {bright.size}")
        w, h = dark.size; ps = self.patch_size
        if self.augment:
            if w < ps or h < ps:
                nw, nh = max(w, ps), max(h, ps)
                dark = TF.resize(dark, (nh, nw)); bright = TF.resize(bright, (nh, nw))
                w, h = dark.size
            left = random.randint(0, w - ps); top = random.randint(0, h - ps)
            dark = TF.crop(dark, top, left, ps, ps); bright = TF.crop(bright, top, left, ps, ps)
            if random.random() < .5: dark, bright = TF.hflip(dark), TF.hflip(bright)
            if random.random() < .5: dark, bright = TF.vflip(dark), TF.vflip(bright)
            if random.random() < .25:
                angle = random.choice((90, 180, 270)); dark, bright = TF.rotate(dark, angle), TF.rotate(bright, angle)
        return self.to_tensor(dark), self.to_tensor(bright)
