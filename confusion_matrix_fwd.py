#!/usr/bin/env python3
"""
Confusion matrix for SpikingResNet LIF model on FWDLDM test split.
Usage: uv run confusion_matrix_fwd.py
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.pl_module import SpikingResNetSNN
from datasets.fwd import FWDLDM

CHECKPOINT_PATH = "/raid/home/michael.siegl/projects/SE-adlif/results/hydra/2026-03-30/14-06-05/ckpt/epoch=272-step=2457.ckpt"
DATA_PATH = "/raid/home/michael.siegl/projects/SE-adlif/data"
OUTPUT_PATH = "/raid/home/michael.siegl/projects/SE-adlif/confusion_matrix_FWD_ResNet_LIF.png"
CLASS_NAMES = ["30cmpm", "60cmpm", "120cmpm"]
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def make_eval_transform(spatial_factor: float = 0.15, time_window: int = 5000, min_len: int = 10):
    """Build inference-time transform without random augmentations."""
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


def main():
    print(f"Device: {DEVICE}")

    # Load model
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = SpikingResNetSNN.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
    model.eval()
    model.to(DEVICE)
    print("Model loaded.")

    # Set up dataset (absolute path avoids the hydra.utils.get_original_cwd() call)
    print("Setting up FWDLDM test split...")
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
    # Replace with eval transform (no random flips) before setup() creates the datasets
    datamodule.static_data_transform = make_eval_transform()
    datamodule.setup()

    test_loader = datamodule.test_dataloader()
    print(f"Test set size: {len(datamodule.data_test)} samples")

    # Inference
    all_preds = []
    all_targets = []

    print("Running inference...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs, targets, block_idx = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            block_idx = block_idx.to(DEVICE)

            outputs = model(inputs)  # (batch, T, num_classes)
            outputs_reduce, _, _ = model.process_predictions_and_compute_losses(
                outputs, targets, block_idx
            )

            targets_flat = targets.flatten()
            mask = targets_flat != model.ignore_target_idx
            preds = outputs_reduce.argmax(dim=-1)

            all_preds.append(preds[mask].cpu())
            all_targets.append(targets_flat[mask].cpu())
            print(f"  Batch {i + 1}/{len(test_loader)}")

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    accuracy = (all_preds == all_targets).mean()
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    print(f"Confusion matrix:\n{cm}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        f"ResNet LIF — FWD dataset (test split)\nAccuracy: {accuracy * 100:.2f}%",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
