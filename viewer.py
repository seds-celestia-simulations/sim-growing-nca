#!/usr/bin/env python3
"""
Interactive viewer for a checkpoint trained with growing_nca.py (keep both files in one folder).

  python viewer.py --ckpt regen.pt

  click + drag   erase a disk of cells (watch it regenerate)
  double-click   plant a new seed cell there
  r              reset to a single centre seed
  space          pause / resume
Needs a GUI matplotlib backend (TkAgg / QtAgg), i.e. run it as a local script, not headless.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from growing_nca import NCA, SIZE, make_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--radius", type=float, default=5.0, help="eraser radius in cells")
    ap.add_argument("--steps-per-frame", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.0, help="sensor rotation in radians (Exp. 4)")
    args = ap.parse_args()

    dev = torch.device("cpu")                    # 64x64 grid: CPU is plenty for interactive use
    model = NCA().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    st = {"x": make_seed(1, dev), "down": False, "paused": False, "t": 0}
    yy, xx = np.mgrid[:SIZE, :SIZE]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.ones((SIZE, SIZE, 3)), interpolation="nearest")
    ax.axis("off")

    def erase(e):
        if e.inaxes is not ax or e.xdata is None:
            return
        hole = torch.from_numpy((xx - e.xdata) ** 2 + (yy - e.ydata) ** 2 < args.radius ** 2)
        st["x"] = st["x"] * (~hole).to(st["x"].dtype)

    def on_press(e):
        if e.inaxes is not ax or e.xdata is None:
            return
        if e.dblclick:
            i, j = int(round(e.ydata)), int(round(e.xdata))
            st["x"][:, 3:, i, j] = 1.0           # alpha + hidden channels = 1, RGB = 0
        else:
            st["down"] = True
            erase(e)

    def on_key(e):
        if e.key == "r":
            st["x"], st["t"] = make_seed(1, dev), 0
        elif e.key == " ":
            st["paused"] = not st["paused"]

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", lambda e: st.update(down=False))
    fig.canvas.mpl_connect("motion_notify_event", lambda e: erase(e) if st["down"] else None)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        if not st["paused"]:
            with torch.no_grad():
                for _ in range(args.steps_per_frame):
                    st["x"] = model(st["x"], theta=args.theta)
            st["t"] += args.steps_per_frame
        s = st["x"][0].numpy()
        rgb, a = s[:3].transpose(1, 2, 0), s[3:4].transpose(1, 2, 0)
        im.set_data(np.clip(rgb + (1.0 - np.clip(a, 0, 1)), 0, 1))   # premultiplied RGB over white
        ax.set_title(f"step {st['t']}")
        return [im]

    anim = FuncAnimation(fig, update, interval=30, cache_frame_data=False)  # keep a reference
    plt.show()


if __name__ == "__main__":
    main()
