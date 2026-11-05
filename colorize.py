"""Colorize an image, a folder of images, or a video with a trained generator.

Examples:
    python colorize.py --ckpt runs/exp1/checkpoints/colorizer.pt --input frame.png --output out.png
    python colorize.py --ckpt ... --input frames_bw/ --output colorized/
    python colorize.py --ckpt ... --input clip_bw.mp4 --output clip_color.mp4

For video, it prints the measured throughput (frames per second) at the end.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cyclegan import Generator
from cyclegan.data import IMAGE_EXTS, list_images, to_gray3
from cyclegan.utils import denorm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_generator(path, device):
    ckpt = torch.load(path, map_location=device)
    G = Generator(n_blocks=ckpt.get("n_blocks", 9)).to(device).eval()
    G.load_state_dict(ckpt["G"])
    return G, ckpt.get("image_size", 256)


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    img = to_gray3(img).resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    return x.permute(2, 0, 1).unsqueeze(0)


def postprocess(y: torch.Tensor, orig_size) -> Image.Image:
    arr = (denorm(y[0]).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize(orig_size, Image.BICUBIC)


@torch.no_grad()
def colorize_image(G, img: Image.Image, size: int, device) -> Image.Image:
    return postprocess(G(preprocess(img, size).to(device)), img.size)


def run_video(G, src: Path, dst: Path, size: int, device):
    import cv2  # only needed for video

    cap = cv2.VideoCapture(str(src))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 24
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    frames, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = colorize_image(G, img, size, device)
        writer.write(cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR))
        frames += 1
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    cap.release(); writer.release()
    print(f"{frames} frames in {elapsed:.1f}s -> {frames / max(elapsed, 1e-9):.1f} FPS on {device} at {size}px")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    G, size = load_generator(args.ckpt, device)
    src, dst = Path(args.input), Path(args.output)

    if src.suffix.lower() in VIDEO_EXTS:
        run_video(G, src, dst, size, device)
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in list_images(src):
            colorize_image(G, Image.open(f).convert("RGB"), size, device).save(dst / f.name)
        print(f"Wrote colorized images to {dst}")
    elif src.suffix.lower() in IMAGE_EXTS:
        colorize_image(G, Image.open(src).convert("RGB"), size, device).save(dst)
        print(f"Wrote {dst}")
    else:
        raise SystemExit(f"Unsupported input: {src}")


if __name__ == "__main__":
    main()
