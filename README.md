# Growing Neural Cellular Automata

A compact PyTorch implementation inspired by [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). It supports training for growth, persistence, and regeneration, along with a regeneration evaluation, hidden-channel probes, and an interactive viewer. This is an independent implementation, not an exact reproduction of the original code.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training uses CUDA when available and otherwise runs on CPU. For an accelerated PyTorch build, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your platform.

## Usage

Train on an image (the included `image.jpg` can be used as a target):

```bash
python growing_nca.py train --target image.jpg --mode regen --out nca.pt
```

The modes are `grow`, `persist`, and `regen`. Use `--iters`, `--batch`, and `--pool` to adjust training; run `python growing_nca.py train --help` for defaults. Training can be computationally expensive. Targets are resized to a square; an image with transparency gives the automaton a transparent background.

Evaluate or inspect a trained checkpoint:

```bash
python growing_nca.py eval --ckpt nca.pt --target image.jpg
python growing_nca.py probe --ckpt nca.pt --target image.jpg
python viewer.py --ckpt nca.pt
```

The probe writes `channels.png`. In the viewer, drag to erase cells, double-click to plant a seed, press `r` to reset, and press space to pause or resume. The viewer requires a graphical Matplotlib backend and a desktop session.

Run the standalone NumPy rotation experiment with `python equiv_test.py`. It prints numerical errors for different sensor rotations; it is a separate forward-pass experiment rather than a test of the PyTorch model.
