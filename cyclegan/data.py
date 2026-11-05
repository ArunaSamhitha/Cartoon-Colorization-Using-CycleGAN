"""Unpaired grayscale <-> color dataset built from a folder of cartoon frames.

CycleGAN does not need matched pairs. We split the color frames into two
disjoint halves: half A is converted to grayscale (domain A), half B stays in
color (domain B). The model never sees the color version of an A image, so it
has to learn colorization from the distribution of B, not by memorizing.
"""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def to_gray3(img: Image.Image) -> Image.Image:
    """Grayscale but kept as 3 channels, so both domains share one shape."""
    return img.convert("L").convert("RGB")


def build_transform(image_size: int, train: bool = True):
    steps = []
    if train:
        steps += [
            transforms.Resize(int(image_size * 1.12), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps += [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
    steps += [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]  # -> [-1, 1]
    return transforms.Compose(steps)


class UnpairedColorizationDataset(Dataset):
    def __init__(self, frames_dir: str | Path, image_size: int = 256, seed: int = 0, train: bool = True):
        files = list_images(frames_dir)
        rng = random.Random(seed)
        rng.shuffle(files)
        half = len(files) // 2
        self.files_a = files[:half]  # shown to the model only in grayscale
        self.files_b = files[half:]  # shown to the model only in color
        if not self.files_a or not self.files_b:
            raise ValueError("Need at least 2 images to build two domains.")
        self.transform = build_transform(image_size, train)

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx):
        a = to_gray3(Image.open(self.files_a[idx % len(self.files_a)]).convert("RGB"))
        # Random B sample so A/B pairings change every epoch (they are unpaired anyway)
        b = Image.open(random.choice(self.files_b)).convert("RGB")
        return {"A": self.transform(a), "B": self.transform(b)}
