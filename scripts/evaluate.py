"""Evaluate colorizers on held-out frames against their true colors.

The held-out frames come from an episode the model never trained on, and we still
have their original colors, so every output can be scored against ground truth.

    python scripts/evaluate.py --frames data/heldout --train-frames data/frames \
        --model full=runs/exp1/checkpoints/colorizer.pt \
        --model no_identity=runs/no_id/checkpoints/colorizer.pt \
        --out results/eval

Methods compared:
  grayscale     the black-and-white input itself (lower bound)
  mean_tint     the input's lightness with the training set's average color added
                (a trivial colorizer the model has to beat; needs --train-frames)
  <name>        each --model checkpoint

Metrics (arrows show which direction is better):
  PSNR ↑, SSIM ↑         pixel agreement with the true colors
  ΔE2000 ↓               mean CIEDE2000 color difference (perceptual color error)
  LPIPS ↓                learned perceptual distance (AlexNet)
  FID ↓                  distribution distance between outputs and real color frames
  Colorfulness           Hasler & Süsstrunk (2003); compare with the "ground truth" row

PSNR and SSIM reward washed-out colors, because the luminance already matches.
Read them together with ΔE2000, LPIPS, FID and colorfulness.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorize import colorize_image, load_generator  # noqa: E402
from cyclegan.data import list_images, to_gray3  # noqa: E402


# ---------- metrics ----------

def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness of a uint8 RGB image."""
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> float 1x3xHxW in [0, 1]."""
    return torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0):
    """Mean and 95% bootstrap confidence interval over frames."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


class Scorer:
    """Per-frame metrics, plus FID accumulated across all frames of one method."""

    def __init__(self, device: str, use_lpips: bool, use_fid: bool):
        from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
        self.psnr, self.ssim = peak_signal_noise_ratio, structural_similarity_index_measure
        self.device = device
        self.lpips = None
        if use_lpips:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        self.use_fid = use_fid

    def new_fid(self):
        if not self.use_fid:
            return None
        from torchmetrics.image.fid import FrechetInceptionDistance
        return FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

    @torch.no_grad()
    def frame(self, pred: np.ndarray, target: np.ndarray) -> dict:
        p, t = to_tensor(pred), to_tensor(target)
        row = {
            "psnr": float(self.psnr(p, t, data_range=1.0)),
            "ssim": float(self.ssim(p, t, data_range=1.0)),
            "delta_e2000": float(deltaE_ciede2000(rgb2lab(target), rgb2lab(pred)).mean()),
            "colorfulness": colorfulness(pred),
        }
        if self.lpips is not None:
            row["lpips"] = float(self.lpips(p.to(self.device), t.to(self.device)))
        return row


# ---------- baselines ----------

def mean_ab(train_dir: str, n: int = 300, width: int = 256, seed: int = 0) -> tuple[float, float]:
    """Average a*/b* chroma over a sample of training frames."""
    files = list_images(train_dir)
    random.Random(seed).shuffle(files)
    a_sum = b_sum = count = 0.0
    for f in files[:n]:
        img = Image.open(f).convert("RGB")
        img = img.resize((width, max(1, round(img.height * width / img.width))), Image.BICUBIC)
        lab = rgb2lab(np.asarray(img))
        a_sum += lab[..., 1].sum(); b_sum += lab[..., 2].sum(); count += lab[..., 0].size
    return a_sum / count, b_sum / count


def tint(gray: Image.Image, ab: tuple[float, float]) -> np.ndarray:
    lab = rgb2lab(np.asarray(gray))
    lab[..., 1], lab[..., 2] = ab
    return (np.clip(lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="Held-out COLOR frames (from an episode not used in training)")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=CKPT",
                    help="A checkpoint to evaluate; repeat for several")
    ap.add_argument("--train-frames", help="Training frames, for the mean_tint baseline")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--width", type=int, default=256, help="Evaluate at this width (aspect ratio kept)")
    ap.add_argument("--limit", type=int, help="Use at most this many held-out frames")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--no-fid", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = list_images(args.frames)
    if args.limit:
        files = files[: args.limit]
    targets = []
    for f in files:
        img = Image.open(f).convert("RGB")
        h = max(1, round(img.height * args.width / img.width))
        targets.append(img.resize((args.width, h), Image.BICUBIC))
    grays = [to_gray3(t) for t in targets]
    print(f"{len(targets)} held-out frames at {args.width}px wide | device: {device}")

    # Each method is a function: grayscale PIL image -> uint8 RGB array.
    methods = {"grayscale": lambda g: np.asarray(g)}
    if args.train_frames:
        ab = mean_ab(args.train_frames)
        print(f"mean_tint baseline uses training-set average a*={ab[0]:.2f}, b*={ab[1]:.2f}")
        methods["mean_tint"] = lambda g, ab=ab: tint(g, ab)
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model needs NAME=CKPT, got {spec!r}")
        G, size = load_generator(path, device)
        methods[name] = lambda g, G=G, size=size: np.asarray(colorize_image(G, g, size, device))

    scorer = Scorer(device, use_lpips=not args.no_lpips, use_fid=not args.no_fid)
    per_frame, summary = [], {}

    # Ground-truth colorfulness, as the reference for the colorfulness column
    gt_color = [colorfulness(np.asarray(t)) for t in targets]
    summary["ground truth"] = {"colorfulness": bootstrap_ci(gt_color)}

    for name, fn in methods.items():
        fid = scorer.new_fid()
        rows, t0 = [], time.perf_counter()
        for f, target, gray in zip(files, targets, grays):
            pred = fn(gray)
            tgt = np.asarray(target)
            row = scorer.frame(pred, tgt)
            rows.append(row)
            per_frame.append({"method": name, "frame": f.name, **row})
            if fid is not None:
                fid.update(to_tensor(tgt).to(device), real=True)
                fid.update(to_tensor(pred).to(device), real=False)
        stats = {k: bootstrap_ci([r[k] for r in rows]) for k in rows[0]}
        if fid is not None:
            stats["fid"] = float(fid.compute())
        summary[name] = stats
        print(f"  {name}: done in {time.perf_counter() - t0:.1f}s")

    # ---- write results ----
    meta = {"held_out_frames": len(targets), "width": args.width, "device": device,
            "models": dict(s.partition("=")[::2] for s in args.model)}
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2))
    with open(out / "per_frame.jsonl", "w") as fh:
        for r in per_frame:
            fh.write(json.dumps(r) + "\n")

    cols = [("psnr", "PSNR ↑", 2), ("ssim", "SSIM ↑", 3), ("delta_e2000", "ΔE2000 ↓", 2),
            ("lpips", "LPIPS ↓", 3), ("fid", "FID ↓", 1), ("colorfulness", "Colorfulness", 1)]
    cols = [c for c in cols if any(c[0] in s for s in summary.values())]
    lines = ["| Method | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, stats in summary.items():
        cells = []
        for key, _, digits in cols:
            v = stats.get(key)
            if v is None:
                cells.append("")
            elif isinstance(v, tuple | list):
                cells.append(f"{v[0]:.{digits}f} ({v[1]:.{digits}f}–{v[2]:.{digits}f})")
            else:
                cells.append(f"{v:.{digits}f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    note = (f"\n{len(targets)} held-out frames, {args.width}px wide. Ranges are 95% bootstrap confidence intervals. "
            "FID on a few hundred frames is biased upward; compare it across rows, not with published numbers.\n")
    table = "\n".join(lines) + "\n" + note
    (out / "results.md").write_text(table)
    print("\n" + table)
    print(f"Wrote {out / 'metrics.json'}, {out / 'results.md'}, {out / 'per_frame.jsonl'}")


if __name__ == "__main__":
    main()
