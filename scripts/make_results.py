"""Build a results grid from held-out frames: grayscale input | colorized | original.

    python scripts/make_results.py --ckpt runs/exp1/checkpoints/colorizer.pt \
        --frames data/heldout --out assets/results.png --n 4
"""
import argparse
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorize import colorize_image, load_generator  # noqa: E402
from cyclegan.data import list_images, to_gray3  # noqa: E402

import torch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", required=True, help="Held-out color frames the model never trained on")
    p.add_argument("--out", default="assets/results.png")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--width", type=int, default=320, help="Width of each tile")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    G, size = load_generator(args.ckpt, device)
    files = list_images(args.frames)
    random.Random(args.seed).shuffle(files)

    rows = []
    for f in files[: args.n]:
        orig = Image.open(f).convert("RGB")
        h = round(orig.height * args.width / orig.width)
        orig = orig.resize((args.width, h), Image.BICUBIC)
        gray = to_gray3(orig)
        color = colorize_image(G, gray, size, device)
        rows.append((gray, color, orig))

    w, h = args.width, rows[0][0].height
    pad = 6
    grid = Image.new("RGB", (3 * w + 4 * pad, len(rows) * (h + pad) + pad), "white")
    for r, tiles in enumerate(rows):
        for c, tile in enumerate(tiles):
            grid.paste(tile.resize((w, h)), (pad + c * (w + pad), pad + r * (h + pad)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Saved {args.out}  (columns: grayscale input | colorized | original)")


if __name__ == "__main__":
    main()
