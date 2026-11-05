"""Train a CycleGAN that maps grayscale cartoon frames (A) to color (B).

Usage:
    python train.py --frames data/frames --epochs 100 --out runs/exp1
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cyclegan import Discriminator, Generator, init_weights
from cyclegan.data import UnpairedColorizationDataset
from cyclegan.utils import ImageBuffer, linear_decay, save_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="Folder of color cartoon frames")
    p.add_argument("--out", default="runs/exp", help="Where to write checkpoints and samples")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--decay-start", type=int, default=50, help="Epoch at which LR starts decaying")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=0.5, help="Fraction of lambda-cycle")
    p.add_argument("--n-blocks", type=int, default=9, help="Residual blocks (6 is fine for 128px)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", help="Path to a checkpoint to continue from")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = UnpairedColorizationDataset(args.frames, args.image_size, seed=args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=device == "cuda")
    print(f"Domain A (grayscale): {len(data.files_a)} frames | Domain B (color): {len(data.files_b)} frames | device: {device}")

    # G: gray -> color, F: color -> gray. D_A judges grayscale, D_B judges color.
    G = Generator(n_blocks=args.n_blocks).to(device)
    F = Generator(n_blocks=args.n_blocks).to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    for net in (G, F, D_A, D_B):
        net.apply(init_weights)

    opt_g = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=args.lr, betas=(0.5, 0.999))
    schedule = linear_decay(args.epochs, args.decay_start)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, schedule)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, schedule)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        for name, net in (("G", G), ("F", F), ("D_A", D_A), ("D_B", D_B)):
            net.load_state_dict(ckpt[name])
        opt_g.load_state_dict(ckpt["opt_g"]); opt_d.load_state_dict(ckpt["opt_d"])
        sched_g.load_state_dict(ckpt["sched_g"]); sched_d.load_state_dict(ckpt["sched_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Least-squares GAN loss: more stable gradients than the log loss.
    gan_loss = nn.MSELoss()
    l1 = nn.L1Loss()
    buffer_a, buffer_b = ImageBuffer(), ImageBuffer()
    history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        totals = {"G": 0.0, "cycle": 0.0, "identity": 0.0, "D": 0.0}
        for step, batch in enumerate(loader):
            real_a, real_b = batch["A"].to(device), batch["B"].to(device)

            # ---- Generators ----
            opt_g.zero_grad(set_to_none=True)
            fake_b = G(real_a)  # colorized
            fake_a = F(real_b)  # desaturated
            pred_fb, pred_fa = D_B(fake_b), D_A(fake_a)
            loss_adv = gan_loss(pred_fb, torch.ones_like(pred_fb)) + gan_loss(pred_fa, torch.ones_like(pred_fa))

            rec_a, rec_b = F(fake_b), G(fake_a)
            loss_cycle = (l1(rec_a, real_a) + l1(rec_b, real_b)) * args.lambda_cycle

            # Identity: G given a color image should leave it alone, and vice versa.
            loss_id = (l1(G(real_b), real_b) + l1(F(real_a), real_a)) * args.lambda_cycle * args.lambda_identity

            loss_g = loss_adv + loss_cycle + loss_id
            loss_g.backward()
            opt_g.step()

            # ---- Discriminators ----
            opt_d.zero_grad(set_to_none=True)
            loss_d = 0.0
            for D, real, fake in ((D_A, real_a, buffer_a.query(fake_a)), (D_B, real_b, buffer_b.query(fake_b))):
                pred_r, pred_f = D(real), D(fake.detach())
                loss_d = loss_d + 0.5 * (gan_loss(pred_r, torch.ones_like(pred_r)) + gan_loss(pred_f, torch.zeros_like(pred_f)))
            loss_d.backward()
            opt_d.step()

            totals["G"] += loss_adv.item(); totals["cycle"] += loss_cycle.item()
            totals["identity"] += loss_id.item(); totals["D"] += loss_d.item()

            if step == 0:
                with torch.no_grad():
                    save_grid(real_a, fake_b, rec_a, out / "samples" / f"epoch_{epoch:03d}.png")

        sched_g.step(); sched_d.step()
        n = len(loader)
        record = {"epoch": epoch, **{k: round(v / n, 4) for k, v in totals.items()},
                  "lr": sched_g.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        history.append(record)
        print(json.dumps(record))

        torch.save({"epoch": epoch, "G": G.state_dict(), "F": F.state_dict(), "D_A": D_A.state_dict(),
                    "D_B": D_B.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "sched_g": sched_g.state_dict(), "sched_d": sched_d.state_dict(), "args": vars(args)},
                   out / "checkpoints" / "last.pt")
        # Small file with only the colorizer, for inference
        torch.save({"G": G.state_dict(), "n_blocks": args.n_blocks, "image_size": args.image_size},
                   out / "checkpoints" / "colorizer.pt")

    with open(out / "history.json", "a") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
