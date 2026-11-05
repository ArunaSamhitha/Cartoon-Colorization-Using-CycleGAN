"""Training helpers: image history buffer, LR schedule, image saving."""
import random

import torch
from torchvision.utils import save_image


class ImageBuffer:
    """Stores up to `size` past generated images.

    When updating the discriminator, half the time we show it an older fake
    instead of the newest one. This stops the generator and discriminator from
    chasing each other in circles (Shrivastava et al., 2017).
    """

    def __init__(self, size: int = 50):
        self.size = size
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return batch
        out = []
        for img in batch.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() < 0.5:
                i = random.randrange(self.size)
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def linear_decay(total_epochs: int, decay_start: int):
    """Constant LR, then linear decay to 0 over the remaining epochs."""

    def factor(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, total_epochs - decay_start))

    return factor


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1) / 2


def save_grid(real_a, fake_b, rec_a, path, nrow: int = 4):
    """Rows: grayscale input | colorized | reconstructed grayscale."""
    n = min(nrow, real_a.size(0))
    grid = torch.cat([real_a[:n], fake_b[:n], rec_a[:n]], 0)
    save_image(denorm(grid), path, nrow=n)
