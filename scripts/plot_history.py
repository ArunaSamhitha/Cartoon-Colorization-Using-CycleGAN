"""Plot training losses from one or more runs' history.json files.

    python scripts/plot_history.py --run full=runs/exp1 --run no_identity=runs/no_id --out assets/loss_curves.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def load(run_dir: str) -> list[dict]:
    rows = {}
    for line in Path(run_dir, "history.json").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["epoch"]] = r  # a resumed run may repeat an epoch; keep the latest
    return [rows[k] for k in sorted(rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--out", default="assets/loss_curves.png")
    args = ap.parse_args()

    keys = [("G", "Generator adversarial"), ("D", "Discriminator"), ("cycle", "Cycle consistency")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4))
    for spec in args.run:
        name, _, path = spec.partition("=")
        hist = load(path)
        epochs = [r["epoch"] + 1 for r in hist]
        for ax, (k, _) in zip(axes, keys):
            ax.plot(epochs, [r[k] for r in hist], label=name, linewidth=1.8)
    for ax, (_, title) in zip(axes, keys):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
