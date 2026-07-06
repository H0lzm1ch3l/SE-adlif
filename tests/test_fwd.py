import os
import sys
import tempfile
import numpy as np
import torch
import tonic

# Ensure project root is importable from this test location.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.fwd import EventDatasetFolder, FWDLDM


def make_fake_event_data(base_dir):
    # 2 classes with one sample each
    for label_name in ["class0", "class1"]:
        cls_dir = os.path.join(base_dir, label_name)
        os.makedirs(cls_dir, exist_ok=True)
        events = np.zeros(2, dtype=[('x', 'i4'), ('y', 'i4'), ('t', 'f4'), ('p', 'i4')])
        events['x'] = [10, 15]
        events['y'] = [20, 25]
        events['t'] = [0.0, 1.0]
        events['p'] = [1, 0]
        np.save(os.path.join(cls_dir, f"{label_name}_sample.npy"), events)


def test_event_dataset_folder_getitem():
    with tempfile.TemporaryDirectory() as td:
        make_fake_event_data(td)

        ds = EventDatasetFolder(
            root=td,
            loader=np.load,
            extensions=".npy",
            transform=lambda ev: torch.from_numpy(tonic.transforms.ToFrame(sensor_size=(320, 320, 2), time_window=5)(ev)).float(),
            ignore_first_timesteps=1,
        )

        sample, label, block_idx = ds[0]

        assert isinstance(sample, torch.Tensor)
        assert sample.ndim == 4
        assert sample.shape[1:] == (2, 320, 320)
        assert label in (0, 1)
        assert block_idx.shape[0] == sample.shape[0]
        assert block_idx[0] == 0
        assert torch.all(block_idx[1:] == 1)


def test_fwldm_setup_and_dataloader(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        fwd_root = os.path.join(td, "FWD")
        make_fake_event_data(fwd_root)

        # This dataset has 2 examples, so override split logic used in FWDLDM.setup.
        def fake_split(indices, test_size, random_state, stratify=None):
            if isinstance(test_size, float):
                test_size = max(1, int(len(indices) * test_size))

            # tiny dataset edge-cases for repeated splits in setup
            if len(indices) <= 2:
                if len(indices) == 0:
                    return [], []
                if len(indices) == 1:
                    return indices, []
                # keep one sample for train, one for test/val
                return [indices[0]], [indices[1]]

            if isinstance(test_size, int):
                test_size = min(test_size, max(1, len(indices) - 1))

            first = indices[: len(indices) - test_size]
            second = indices[len(indices) - test_size :]
            return first, second

        monkeypatch.setattr("datasets.fwd.train_test_split", fake_split)

        dm = FWDLDM(
            data_path=td,
            spatial_factor=1.0,
            time_factor=1.0,
            time_window=5,
            batch_size=2,
            num_workers=0,
            pad_to_min_size=1,
            num_classes=2,
            ignore_first_timesteps=1,
        )
        dm.prepare_data()
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert len(batch) >= 2
        x, y, block = batch[0], batch[1], batch[2]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(block, torch.Tensor)
        assert x.shape[0] <= 2
        assert y.shape[0] <= 2
        assert block.shape[0] <= 2
