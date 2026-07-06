#!/usr/bin/env python3
"""
Temporal distribution of output activations per class for SpikingResNet LIF on FWD.

For each true class, plots the mean ± std of the softmax output for every
class neuron over time, revealing when the network's confidence builds up.

Usage: uv run spike_timing_distribution.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import tonic
from tonic.transforms import ToFrame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models.pl_module import SpikingResNetSNN
from datasets.fwd import FWDLDM

CHECKPOINT_PATH = "/raid/home/michael.siegl/projects/SE-adlif/results/hydra/2026-03-30/14-06-05/ckpt/epoch=272-step=2457.ckpt"
DATA_PATH = "/raid/home/michael.siegl/projects/SE-adlif/data"
OUTPUT_PATH = "/raid/home/michael.siegl/projects/SE-adlif/spike_timing_distribution.png"
CLASS_NAMES = ["30cmpm", "60cmpm", "120cmpm"]
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


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


def collect_output_sequences(model, test_loader, device):
    """
    Returns:
        sequences: list of (T, num_classes) softmax tensors, one per sample
        true_labels: list of int, one per sample
        seq_lengths: list of int (actual non-padded length from block_idx)
    """
    sequences = []
    true_labels = []
    seq_lengths = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets, block_idx = batch
            inputs = inputs.to(device)
            block_idx = block_idx.to(device)

            # Raw output sequence: (batch, T, num_classes)
            outputs = model(inputs)
            softmax_out = torch.softmax(outputs, dim=-1).cpu()  # (batch, T, num_classes)

            # True label per sample: targets has shape (batch, 2) after PadTensors;
            # column 0 is the dummy ignore block, column 1 is the real label
            labels = targets[:, 1].cpu()  # (batch,)

            # Actual sequence length = number of timesteps where block_idx == 1
            lengths = (block_idx == 1).sum(dim=1).cpu()  # (batch,)

            for i in range(softmax_out.shape[0]):
                T_actual = lengths[i].item()
                # Keep only the non-padding, non-ignore timesteps (block_idx==1)
                mask = (block_idx[i] == 1).cpu()
                seq = softmax_out[i][mask]  # (T_actual, num_classes)
                sequences.append(seq)
                true_labels.append(labels[i].item())
                seq_lengths.append(T_actual)

    return sequences, true_labels, seq_lengths


def interpolate_to_length(seq: torch.Tensor, target_len: int) -> np.ndarray:
    """Linearly interpolate a (T, C) sequence to target_len timesteps."""
    T, C = seq.shape
    if T == target_len:
        return seq.numpy()
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    out = np.stack(
        [np.interp(x_new, x_old, seq[:, c].numpy()) for c in range(C)],
        axis=1,
    )
    return out  # (target_len, C)


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

    print("Collecting output sequences...")
    sequences, true_labels, seq_lengths = collect_output_sequences(
        model, datamodule.test_dataloader(), DEVICE
    )

    print(f"Collected {len(sequences)} sequences")
    print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
          f"median={int(np.median(seq_lengths))}")

    # Normalise all sequences to a common relative time axis (100 points = 0–100%)
    N_POINTS = 100
    num_classes = sequences[0].shape[1]

    # per_class[c] → list of (N_POINTS, num_classes) arrays for true class c
    per_class = {c: [] for c in range(num_classes)}
    for seq, label in zip(sequences, true_labels):
        interp = interpolate_to_length(seq, N_POINTS)
        per_class[int(label)].append(interp)

    # Stack: per_class[c] → (n_samples, N_POINTS, num_classes)
    per_class = {c: np.stack(v, axis=0) for c, v in per_class.items()}

    # ── Figure layout: 3 rows (true classes) × 1 column ───────────────────────
    colors = ["#2196F3", "#FF9800", "#4CAF50"]   # one colour per output neuron
    t_axis = np.linspace(0, 100, N_POINTS)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        "Temporal output distribution per class — ResNet LIF on FWD\n"
        "(softmax probability over normalised sequence length)",
        fontsize=13,
    )

    for row, true_cls in enumerate(range(num_classes)):
        ax = axes[row]
        data = per_class[true_cls]  # (n_samples, N_POINTS, num_classes)
        n = data.shape[0]

        for out_cls in range(num_classes):
            vals = data[:, :, out_cls]           # (n_samples, N_POINTS)
            mean = vals.mean(axis=0)
            std  = vals.std(axis=0)

            lw = 2.5 if out_cls == true_cls else 1.2
            ls = "-"  if out_cls == true_cls else "--"
            alpha_band = 0.25 if out_cls == true_cls else 0.10

            ax.plot(t_axis, mean, color=colors[out_cls], lw=lw, ls=ls,
                    label=CLASS_NAMES[out_cls])
            ax.fill_between(t_axis, mean - std, mean + std,
                            color=colors[out_cls], alpha=alpha_band)

        ax.set_ylabel("Softmax probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"True class: {CLASS_NAMES[true_cls]}  (n={n})", fontsize=11)
        ax.axhline(1 / num_classes, color="grey", lw=0.8, ls=":", label="chance")
        ax.legend(loc="upper left", fontsize=9, ncol=4)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Relative sequence position (%)", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
