"""Measure colorizer throughput across batch sizes and precisions.

    python scripts/benchmark.py --ckpt runs/exp1/checkpoints/colorizer.pt --out results/benchmark.json

Times only the generator's forward pass on the GPU (no video decoding or file I/O),
after warm-up, so the numbers show what the model itself can sustain.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colorize import load_generator  # noqa: E402


@torch.no_grad()
def measure(G, size, batch, dtype, device, iters, warmup):
    x = torch.randn(batch, 3, size, size, device=device)
    use_amp = dtype is not None
    ctx = (lambda: torch.autocast(device, dtype=dtype)) if use_amp else torch.no_grad
    for _ in range(warmup):
        with ctx():
            G(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        with ctx():
            G(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    row = {"batch": batch, "precision": str(dtype).replace("torch.", "") if dtype else "float32",
           "frames_per_s": round(batch * iters / elapsed, 1),
           "ms_per_batch": round(1000 * elapsed / iters, 2)}
    if device == "cuda":
        row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", default="1,4,16,32")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    G, size = load_generator(args.ckpt, device)
    G.eval()

    precisions = [None]
    if device == "cuda":
        precisions.append(torch.float16)
        if torch.cuda.get_device_capability()[0] >= 8:
            precisions.append(torch.bfloat16)

    rows = []
    for dtype in precisions:
        for b in (int(s) for s in args.batches.split(",")):
            try:
                rows.append(measure(G, size, b, dtype, device, args.iters, args.warmup))
            except torch.OutOfMemoryError:
                rows.append({"batch": b, "precision": str(dtype), "error": "out of memory"})
                torch.cuda.empty_cache()
            print(json.dumps(rows[-1]))

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    ok = [r for r in rows if "frames_per_s" in r]
    base = next((r for r in ok if r["batch"] == 1 and r["precision"] == "float32"), None)
    best = max(ok, key=lambda r: r["frames_per_s"])
    result = {"device": gpu, "image_size": size, "rows": rows, "best": best,
              "speedup_vs_batch1_fp32": round(best["frames_per_s"] / base["frames_per_s"], 1) if base else None}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n{gpu} @ {size}px: best {best['frames_per_s']} frames/s "
          f"(batch {best['batch']}, {best['precision']}), {result['speedup_vs_batch1_fp32']}x over batch-1 float32")


if __name__ == "__main__":
    main()
