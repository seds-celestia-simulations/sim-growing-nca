#!/usr/bin/env python3
"""
Compact PyTorch reimplementation of "Growing Neural Cellular Automata"
(Mordvintsev, Randazzo, Niklasson, Levin; Distill 2020), plus tools the article
does not include: a quantitative regeneration benchmark and hidden-channel probes.

  python growing_nca.py train    --target lizard.png --mode regen   --out regen.pt
  python growing_nca.py eval     --ckpt regen.pt --target lizard.png
  python growing_nca.py probe    --ckpt regen.pt --target lizard.png

Training modes map onto the article's experiments:
  grow     Exp. 1  fresh seed every step, loss at a random T in [64, 96]
  persist  Exp. 2  sample pool (+ reseeding the worst sample)
  regen    Exp. 3  sample pool + circular damage on the 3 lowest-loss samples

NOTE: written against the article's description and the usual PyTorch API; the
reference Colab differs in small ways (e.g. it masks living cells both before
and after the update, and uses a different target preprocessing).
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

C, SIZE = 16, 128          # channels per cell, grid side length


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def alive(x):
    """(B,1,H,W) float mask: 1 where some cell in the 3x3 neighbourhood has alpha > 0.1."""
    return (F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1).to(x.dtype)


class NCA(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0   # Sobel (scaled)
        ident = torch.zeros(3, 3)
        ident[1, 1] = 1.0
        self.register_buffer("sx", sx)
        self.register_buffer("ident", ident)
        self.fc1 = nn.Conv2d(3 * C, hidden, 1)      # 48 -> 128   (1x1 conv == per-cell dense)
        self.fc2 = nn.Conv2d(hidden, C, 1)          # 128 -> 16
        nn.init.zeros_(self.fc2.weight)             # "do-nothing" initial dynamics
        nn.init.zeros_(self.fc2.bias)

    def kernels(self, theta=0.0):
        """(3C,1,3,3) depthwise kernels; output channel 3c+j is kernel j on input channel c
        with j in (identity, d/dx', d/dy'), where (x', y') is the sensor frame rotated by theta."""
        c, s = math.cos(theta), math.sin(theta)
        sy = self.sx.T
        kx = c * self.sx - s * sy                   # [Kx; Ky] = R(theta) [Sobel_x; Sobel_y]
        ky = s * self.sx + c * sy
        k = torch.stack([self.ident, kx, ky])       # (3,3,3)
        return k.repeat(C, 1, 1).unsqueeze(1)       # (3C,1,3,3)

    def forward(self, x, fire_rate=0.5, theta=0.0):
        p = F.conv2d(x, self.kernels(theta).to(x.dtype), padding=1, groups=C)  # perception
        ds = self.fc2(F.relu(self.fc1(p)))                                      # residual update
        if fire_rate < 1.0:                                                     # async updates
            ds = ds * (torch.rand_like(x[:, :1]) < fire_rate).to(x.dtype)
        x = x + ds
        return x * alive(x)                                                     # kill dead cells


# --------------------------------------------------------------------------- #
# Data / losses / damage
# --------------------------------------------------------------------------- #
def load_target(path, size=112):
    """RGBA image -> (4,SIZE,SIZE) tensor with premultiplied RGB (so background RGB = 0,
    consistent with dead cells having an all-zero state)."""
    from PIL import Image
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:]
    pad = (SIZE - size) // 2
    a = np.pad(a, ((pad, SIZE - size - pad), (pad, SIZE - size - pad), (0, 0)))
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def make_seed(n, dev):
    x = torch.zeros(n, C, SIZE, SIZE, device=dev)
    x[:, 3:, SIZE // 2, SIZE // 2] = 1.0            # alpha and hidden channels = 1, RGB = 0
    return x


def per_sample_loss(x, target):
    return ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))


def damage(x):
    """Erase one random disk per sample (radius 0.1-0.3 of the grid side)."""
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                            torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
    cy, cx = torch.rand(2, B, 1, 1, device=x.device) * 0.5 * H + 0.25 * H
    r = (torch.rand(B, 1, 1, device=x.device) * 0.2 + 0.1) * H
    hole = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2            # (B,H,W)
    return x * (~hole).unsqueeze(1).to(x.dtype)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = load_target(args.target).to(dev)
    model = NCA().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(
    opt,
    milestones=[max(1, int(args.iters * 0.2))],
    gamma=0.1,
)
    pool = make_seed(args.pool, dev)
    print(f"params: {sum(p.numel() for p in model.parameters())}  device: {dev}")

    for it in range(1, args.iters + 1):
        if args.mode == "grow":
            idx, x = None, make_seed(args.batch, dev)
        else:
            idx = torch.randperm(args.pool, device=dev)[:args.batch]
            x = pool[idx]                                       # advanced indexing -> copy
            order = per_sample_loss(x, target).argsort(descending=True)   # worst first
            x, idx = x[order], idx[order]
            x[0] = make_seed(1, dev)[0]                         # reseed the worst sample
            if args.mode == "regen":
                n_damage = min(3, x.shape[0] - 1)
                if n_damage > 0:
                    x[-n_damage:] = damage(x[-n_damage:])

        T = np.random.randint(64, 97)

        for _ in range(T):
            x = checkpoint(
                model,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        loss = per_sample_loss(x, target).mean()

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():                                   # per-variable L2 grad norm
            for p in model.parameters():
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        if idx is not None:
            pool[idx] = x.detach()                              # write outputs back

        if it % 100 == 0:
            print(f"it {it:5d}  loss {loss.item():.5f}")
        if it % 1000 == 0 or it == args.iters:
            torch.save(model.state_dict(), args.out)
    print("saved", args.out)


# --------------------------------------------------------------------------- #
# Evaluation (the article shows regeneration qualitatively; this quantifies it)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(model, x, steps, **kw):
    for _ in range(steps):
        x = model(x, **kw)
    return x


@torch.no_grad()
def evaluate(model, target, dev, n=16, t_grow=100, t_rec=100, t_long=500):
    x = rollout(model, make_seed(n, dev), t_grow)
    L = lambda s: per_sample_loss(s, target).mean().item()
    print(f"grown   @t={t_grow}: loss {L(x):.5f}")
    print(f"persist @t={t_long}: loss {L(rollout(model, x, t_long - t_grow)):.5f}")
    yy, xx = torch.meshgrid(torch.arange(SIZE, device=dev), torch.arange(SIZE, device=dev), indexing="ij")
    c = SIZE // 2
    holes = {"left half": xx < c, "right half": xx >= c, "top half": yy < c,
             "bottom half": yy >= c, "center 16x16": ((yy - c).abs() < 8) & ((xx - c).abs() < 8)}
    print(f"{'damage':<14}{'loss after hit':>16}{f'loss +{t_rec} steps':>18}")
    for name, hole in holes.items():
        xd = x * (~hole).to(x.dtype)
        l0 = L(xd)
        print(f"{name:<14}{l0:>16.5f}{L(rollout(model, xd, t_rec)):>18.5f}")


# --------------------------------------------------------------------------- #
# Hidden-channel probes
# --------------------------------------------------------------------------- #
def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-3):
    """Ridge regression (unregularised bias); returns per-target R^2 on the test set."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    reg = lam * len(A) * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
    res = ((yte - B @ w) ** 2).sum(0)
    tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return 1 - res / tot


@torch.no_grad()
def probe(model, target, dev, n=32, steps=100, out_png="channels.png"):
    x = rollout(model, make_seed(n, dev), steps)
    xs = x.cpu().numpy()                                        # (n,C,H,W)
    living = xs[:, 3] > 0.1

    # (1) How well do channels linearly decode position? Held-out spatial blocks (8x8
    # checkerboard) so the regressor must interpolate between regions, not memorise cells.
    yy, xx = np.mgrid[:SIZE, :SIZE]
    block = ((yy // 8 + xx // 8) % 2 == 0)
    feats = xs.transpose(0, 2, 3, 1)                            # (n,H,W,C)
    coords = np.stack([yy - SIZE // 2, xx - SIZE // 2], -1).astype(np.float64)
    sets = {"RGBA only": list(range(4)), "hidden only": list(range(4, C)), "RGBA + hidden": list(range(C))}
    print("Linear decoding of (row, col) from per-cell state; R^2 on held-out blocks")
    for name, ch in sets.items():
        tr = np.broadcast_to(block, living.shape) & living
        te = np.broadcast_to(~block, living.shape) & living
        r2 = ridge_r2(feats[tr][:, ch], coords[np.nonzero(tr)[1:]],
                      feats[te][:, ch], coords[np.nonzero(te)[1:]])
        print(f"  {name:<14} R2_row={r2[0]:+.3f}  R2_col={r2[1]:+.3f}")
    print("  (decodability alone cannot separate a positional code from position-correlated\n"
          "   features: the image is non-stationary. The ablation below is the causal test.)")

    # (2) Causal test: clamp one hidden channel to its mean over living cells, every step.
    mean_c = torch.tensor([(xs[:, k][living]).mean() for k in range(C)], device=dev)
    base = per_sample_loss(x, target).mean().item()
    print(f"\nChannel mean-ablation (loss at t={steps}; unablated = {base:.5f})")
    for k in range(4, C):
        z = make_seed(n, dev)
        for _ in range(steps):
            z = model(z)
            z[:, k] = mean_c[k] * alive(z)[:, 0]
        print(f"  hidden ch {k:2d}: loss {per_sample_loss(z, target).mean().item():.5f}")

    # (3) Look at the channels.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 6, figsize=(14, 7))
    rgb = np.clip(xs[0, :3].transpose(1, 2, 0), 0, 1)
    axs[0, 0].imshow(rgb)
    axs[0, 0].set_title("RGB")
    for i, k in enumerate(range(3, C)):
        ax = axs.flat[i + 1]
        v = np.abs(xs[0, k]).max() + 1e-8
        ax.imshow(xs[0, k], cmap="coolwarm", vmin=-v, vmax=v)
        ax.set_title(f"ch {k}" + (" (alpha)" if k == 3 else ""))
    for ax in axs.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print("saved", out_png)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--target", required=True)
    t.add_argument("--mode", choices=["grow", "persist", "regen"], default="regen")
    t.add_argument("--iters", type=int, default=10000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--pool", type=int, default=1024)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="nca.pt")
    for name in ("eval", "probe"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", required=True)
        s.add_argument("--target", required=True)
    args = ap.parse_args()

    if args.cmd == "train":
        train(args)
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCA().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    target = load_target(args.target).to(dev)
    (evaluate if args.cmd == "eval" else probe)(model, target, dev)


if __name__ == "__main__":
    main()
