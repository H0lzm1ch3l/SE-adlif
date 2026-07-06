#!/usr/bin/env python3
"""
Spike timing histogram: input current to the LI output layer per timestep.

Hooks the LI output layer to capture the weighted input current
  current = W @ spikes_256 + b  →  shape (batch, 3)
one value per output neuron per timestep.

Layout: 3 subplots (one per true class), each with 3 grouped bars per timestep
(one per LI output neuron / predicted class) so you can see which class neuron
receives input — and when — for each group of true-class samples.

Usage: uv run spike_histogram.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import tonic
from tonic.transforms import ToFrame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.pl_module import SpikingResNetSNN
from datasets.fwd import FWDLDM

CHECKPOINT_PATH = "/raid/home/michael.siegl/projects/SE-adlif/results/hydra/2026-03-30/14-06-05/ckpt/epoch=272-step=2457.ckpt"
DATA_PATH = "/raid/home/michael.siegl/projects/SE-adlif/data"
OUTPUT_PATH = "/raid/home/michael.siegl/projects/SE-adlif/spike_histogram.png"
CLASS_NAMES = ["30cmpm", "60cmpm", "120cmpm"]
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# one colour per LI output neuron (= predicted class)
NEURON_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]


def make_eval_transform(spatial_factor: float = 0.15, time_window: int = 5000, min_len: int = 10):
    sensor_size = (
        int(math.ceil(320 * spatial_factor)),
        int(math.ceil(320 * spatial_factor)),
        2,
    )
    _to_frame = ToFrame(sensor_size=sensor_size, time_window=time_window)
    event_to_tensor = lambda x: torch.from_numpy(_to_frame(x)).float()

    def pad(x):
        if x.shape[0] < min_len:
            pad_frames = torch.zeros((min_len - x.shape[0], *sensor_size[::-1]))
            x = torch.cat((x, pad_frames), dim=0)
        return x

    return tonic.transforms.Compose([
        tonic.transforms.Downsample(spatial_factor=spatial_factor, time_factor=1),
        event_to_tensor,
        pad,
    ])


def collect_li_inputs(model, test_loader, device):
    """
    Hook the LI output layer to capture the weighted input current at each timestep.

    Returns:
        currents: np.array (n_samples, T, 3)  – input current per LI neuron per timestep
        true_labels: np.array (n_samples,)
    """
    timestep_buffer = []   # list of (batch, 3) tensors, reset per batch

    def hook(module, inp, out):
        x = inp[0]   # (batch, 256) – spike features after global avg pool
        current = F.linear(x, module.weight, module.bias).detach().cpu()  # (batch, 3)
        timestep_buffer.append(current)

    handle = model.model.output_layer.register_forward_hook(hook)

    all_currents = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, block_idx = batch
            inputs = inputs.to(device)

            timestep_buffer.clear()
            _ = model(inputs)   # hook fires T times

            # (T, batch, 3) -> (batch, T, 3)
            currents = torch.stack(timestep_buffer, dim=0).permute(1, 0, 2).numpy()
            labels = targets[:, 1].cpu().numpy()   # col 0 is dummy -1 block

            all_currents.append(currents)
            all_labels.append(labels)

    handle.remove()

    return np.concatenate(all_currents, axis=0), np.concatenate(all_labels)


def main():
    print(f"Device: {DEVICE}")

    print("Loading checkpoint...")
    model = SpikingResNetSNN.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)

    print("Setting up dataset...")
    datamodule = FWDLDM(
        data_path=DATA_PATH,
        spatial_factor=0.15,
        time_factor=1,
        time_window=5000,
        batch_size=32,
        num_workers=8,
        pad_to_min_size=10,
        num_classes=3,
        ignore_first_timesteps=1,
    )
    datamodule.static_data_transform = make_eval_transform()
    datamodule.setup()

    print("Collecting LI input currents...")
    currents, true_labels = collect_li_inputs(
        model, datamodule.test_dataloader(), DEVICE
    )

    n_samples, T, n_neurons = currents.shape
    print(f"Samples: {n_samples}, Timesteps: {T}, LI neurons: {n_neurons}")

    # ── Figure: 3 rows (true classes), all LI neurons overlaid ───────────────
    timesteps = np.arange(1, T + 1)
    bar_width = 0.25
    offsets = np.linspace(-(n_neurons - 1) * bar_width / 2,
                           (n_neurons - 1) * bar_width / 2, n_neurons)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, sharey=True)
    fig.suptitle(
        "Input current to LI output neurons per timestep — ResNet LIF on FWD\n"
        r"current$_j$ = $W_j \cdot$ spikes$_{256}$ + $b_j$  |  mean ± std across samples",
        fontsize=12,
    )

    for cls_idx in range(len(CLASS_NAMES)):
        ax = axes[cls_idx]
        mask = true_labels == cls_idx
        n = mask.sum()

        for neuron_idx in range(n_neurons):
            vals = currents[mask, :, neuron_idx]   # (n_cls, T)
            mean = vals.mean(axis=0)
            std  = vals.std(axis=0)

            x = timesteps + offsets[neuron_idx]
            lw = 2.0 if neuron_idx == cls_idx else 1.0
            ax.bar(x, mean, width=bar_width,
                   color=NEURON_COLORS[neuron_idx], alpha=0.82,
                   label=f"neuron {CLASS_NAMES[neuron_idx]}",
                   linewidth=lw,
                   edgecolor="black" if neuron_idx == cls_idx else "none")
            ax.errorbar(x, mean, yerr=std,
                        fmt="none", ecolor=NEURON_COLORS[neuron_idx],
                        elinewidth=1.0, capsize=2, alpha=0.8)

        ax.set_ylabel("Input current", fontsize=10)
        ax.set_title(f"True class: {CLASS_NAMES[cls_idx]}  (n={n})", fontsize=11)
        ax.set_xticks(timesteps)
        ax.set_xticklabels([f"t{i}" for i in timesteps])
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Timestep", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
