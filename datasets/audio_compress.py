from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from typing import Callable, Optional
import hydra
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import tonic

import torch.utils.data
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from datasets.utils.diskcache import DiskCachedDataset
from datasets.utils.pad_tensors import PadTensors
from pathlib import Path
import shutil
import torchaudio
import pickle
from tqdm import tqdm
import hashlib


def save_chunk_map(chunk_map, filepath):
    """Saves the chunk map to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(chunk_map, f)
    print(f"Chunk map saved to {filepath}")


def load_chunk_map(filepath):
    """Loads the chunk map from a pickle file if it exists."""
    if Path(filepath).is_file():
        with open(filepath, "rb") as f:
            chunk_map = pickle.load(f)
        print(f"Chunk map loaded from {filepath}")
        return chunk_map
    return None


def copy_wave_files_with_path_names(root_path, destination_path):
    root_path = Path(root_path)
    destination_path = Path(destination_path)

    # Ensure destination directory exists
    destination_path.mkdir(parents=True, exist_ok=True)

    # Find all .wav files recursively in the root directory
    for wav_file in root_path.rglob("*.wav"):
        # Get the parts of the path between root_path and the file
        relative_parts = wav_file.relative_to(root_path).parent.parts
        # Generate the new filename: n1_n2_..._originalfilename.wav
        new_name = "_".join(relative_parts) + "_" + wav_file.name
        # Define the full destination path
        destination_file = destination_path / new_name
        # Copy the file to the new location
        shutil.copy2(wav_file, destination_file)


def process_resampling(wav_file, output_path, target_sample_rate):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(wav_file)

    with torch.no_grad():
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    # Define the output file path (same name as the original)
    output_file = output_path / wav_file.name

    # Save the resampled audio to the output path
    torchaudio.save(output_file, waveform, target_sample_rate)


def resample_and_save_wav_files(input_path, output_path, base_freq, new_freq):
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all .wav files to process
    wav_files = list(input_path.rglob("*.wav"))

    # Use multiprocessing to speed up the process
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_resampling, wav_file, output_path, new_freq)
            for wav_file in wav_files
        ]
        # Wait for all futures to complete
        for future in futures:
            future.result()


def create_chunk_map(file_paths, chunk_size):
    chunk_map = {}  # Dictionary to hold the mapping {chunk_id: (file_path, start_sample)}
    chunk_id = 0  # Initialize the global chunk ID
    if chunk_size == -1:
        chunk_map = {i: (v, 0) for i, v in enumerate(file_paths)}
        return chunk_map

    for wav_file in file_paths:
        # Load the audio file to get its number of samples
        waveform, sample_rate = torchaudio.load(wav_file)

        # Calculate the number of valid chunks for this file
        num_samples = waveform.shape[1]
        num_chunks = (
            num_samples // chunk_size
        )  # Integer division to discard incomplete chunks

        # For each chunk in the current file, add an entry to the chunk_map
        for i in range(num_chunks):
            start_sample = i * chunk_size
            chunk_map[chunk_id] = (wav_file, start_sample)
            chunk_id += 1  # Increment the global chunk ID

    return chunk_map


def get_chunk_by_id(chunk_id, chunk_map, chunk_size, norm_func):
    wav_file, start_sample = chunk_map[chunk_id]
    # Load the waveform for this file
    waveform, sample_rate = torchaudio.load(wav_file)
    waveform = norm_func(waveform)
    if chunk_size == -1:
        return waveform
    # Extract the chunk
    end_sample = start_sample + chunk_size
    chunk = waveform[:, start_sample:end_sample]

    return chunk


norm_map = {
    "none": lambda x: x,
    "-1_1": lambda x: -1 + 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x)),
    "0_1": lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)),
    "standard": lambda x: (x - torch.mean(x)) / (torch.var(x) + 1e-9),
}


class LibriTTS(Dataset):
    def __init__(
        self,
        save_to: str,
        cache_path: str = None,
        max_sample: int = -1,
        sampling_freq: int = 24_000,
        sample_length: int = -1,
        normalization: str = "none",
        train: bool = True,
        debug: bool = True,
        transform=None,
        target_transform=None,
        get_metadata: bool = False,
    ):
        # libriTTS stucture {saveto}/user_id/chapter_id
        # 0. check if root/{test|valid|train}/sampling_freq/full dir exist if it does get all paths here and go to 4
        # 1. get all wave file path
        # 2. copy from file from root to root/debug/train/test to root/{test | train | debug}/24_000/full dir
        # 3. if sampling_freq is not 24_000 create a new dir root/debug/train/test to root/{test | train | debug}/{sampling_freq}/full
        #   and resample, return the list of files here
        # 4. if sample_length is different than -1, create a new dir root/debug/train/test to root/{test | train | debug}/{sampling_freq}/{sample_length}
        # and cut each audio as name_i creating new file for each subsequence and dropping non complete subsequence
        # (this consume too much space one could save cut idx from each file and layzy load from full but time constraint)
        # alternativelly one could workout the sequence length from original sampling freq and new freq (original/new * seq_len ?)
        # then resample from full freq, then use the cachedisk to save the results
        try_path = os.path.join(save_to, "dev-clean", "174", "84280")
        if not os.path.exists(try_path):
            print(f"Path {try_path} does not exist")
            raise Exception(f"Path {try_path} does not exist")
        self.transform = transform
        self.target_transform = target_transform
        self.get_metadata = get_metadata
        self.norm_type = normalization
        self.norm_func = norm_map[normalization]
        self.root_dir = Path(save_to)
        self.cache_path = Path(cache_path)
        split = "debug"
        if not debug:
            split = "train" if train else "test"
        full_split = split + "/24000"
        freq_split = split + f"/{sampling_freq}"
        full_split_dir = self.cache_path / full_split
        freq_split_dir = self.cache_path / freq_split
        if (full_split_dir).exists() and any((freq_split_dir).iterdir()):
            print("directory already exist")
        else:
            # for now only get debug
            if debug:
                copy_wave_files_with_path_names(
                    self.root_dir / "dev-clean", full_split_dir
                )
            else:
                raise NotImplementedError("Only debug is implemented for now")
        # 24_000/full split exist

        if sampling_freq != 24_000 and (
            not freq_split_dir.exists() or not any(full_split_dir.iterdir())
        ):
            # resample
            print(f"Resample wave file from 24kHz to {sampling_freq}Hz")
            resample_and_save_wav_files(
                full_split_dir, freq_split_dir, 24_000, sampling_freq
            )
        print(f"Wave files are resampled to {sampling_freq}Hz")
        self.wave_files_path = list(freq_split_dir.rglob("*.wav"))
        # associate a chunk idx with a file_path and starting sample idx
        chunk_map = load_chunk_map(freq_split_dir / f"{sample_length}_map.pkl")
        if chunk_map is None:
            chunk_map = create_chunk_map(self.wave_files_path, sample_length)
            save_chunk_map(chunk_map, freq_split_dir / f"{sample_length}_map.pkl")
        self.chunk_map = chunk_map
        self.num_samples = len(self.chunk_map)
        print(f"Waves files are splited into {sample_length} samples length")
        self.sample_length = sample_length
        self.sampling_freq = sampling_freq
        self.max_sample = max_sample

    def __len__(self):
        return (
            self.num_samples
            if self.max_sample == -1
            else min(self.num_samples, self.max_sample)
        )

    def __getitem__(self, index):
        inputs: torch.Tensor = get_chunk_by_id(
            index, self.chunk_map, self.sample_length, self.norm_func
        ).T
        # inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
        targets = inputs.clone()
        block_idx = torch.ones((inputs.shape[0],), dtype=torch.int32)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.get_metadata:
            return inputs, targets, block_idx, {}
        else:
            return inputs, targets, block_idx


class CompressLibri(pl.LightningDataModule):
    """
    Create data module for the damped oscilator system,
    the parameter of the system are selected randomly
    but fixed for the trial.
    The objective is to predict the next position.
    The generalization is determined by correct prediction from
    unseen initial conditions.
    max_freq determined the maximum frequency that you want to represent
    sensible value would max_freq > 2*pi*sqrt(max(K)/min(M)),
    the maximal possible harmonic frequency of one spring-mass system in isolation.
    where K and M are the spring and mass coefficients.


    """

    def __init__(
        self,
        data_path: str,
        cache_path: str = None,
        max_sample: int = -1,
        sampling_freq: int = 24_000,
        sample_length: int = -1,
        prediction_delay: int = 0,
        zero_input_proba: float = 0,
        batch_size: int = 32,
        num_workers: int = 1,
        fits_into_ram: bool = False,
        normalization: str = "none",
        name: str = None,  # for hydra
        required_model_size: str = None,  # for hydra
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fits_into_ram = fits_into_ram
        self.collate_fn = PadTensors(require_padding=False)
        self.normalization = normalization
        if not os.path.isabs(data_path):
            cwd = hydra.utils.get_original_cwd()
            data_path = os.path.abspath(os.path.join(cwd, data_path))

        self.train_dataset_ = LibriTTS(
            save_to=data_path + "/LibriTTS",
            cache_path=cache_path + "/LibriTTS",
            max_sample=max_sample,
            sampling_freq=sampling_freq,
            sample_length=sample_length,
            normalization=normalization,
            debug=True,
            transform=None,
            target_transform=None,
        )
        self.cache_path = (
            cache_path
            + f"/LibriTTS_hz-{sampling_freq}_sl-{sample_length}_norm-{normalization}"
        )

        def delay_transform(inputs, targets, block_idx):
            # add zero padding to account for possible prediciton delay
            # idea is that it is potentially complex for the model to predict
            # y[t] = L(x[0:t]) where L is the model, x inputs, y the output
            # delay allow predition as y[t - delay] = L([x[0:t]])
            inputs = torch.concatenate(
                (
                    inputs,
                    torch.zeros(
                        (prediction_delay, 1), device=inputs.device, dtype=inputs.dtype
                    ),
                )
            )
            block_idx = torch.ones((inputs.shape[0],), dtype=torch.int32)
            return inputs, targets, block_idx
        def zero_inputs(inputs, targets, block_idx):
            rd = torch.rand(()).item()
            if rd < zero_input_proba:
                inputs = torch.zeros_like(inputs)
                targets = torch.zeros_like(targets)
            return inputs, targets, block_idx
        def compose_full_transform(inputs, targets, block_idx):
            inputs, targets, block_idx = delay_transform(inputs, targets, block_idx)
            return zero_inputs(inputs, targets, block_idx)

        self.train_dataset_ = DiskCachedDataset(
            self.train_dataset_,
            cache_path=self.cache_path,
            full_transform=compose_full_transform
        )

        self.train_dataset_, self.valid_dataset_, self.test_dataset_ = (
            torch.utils.data.random_split(
                self.train_dataset_,
                [0.8, 0.1, 0.1],
                generator=None,
            )
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage == "validate":
            self.data_train = self.train_dataset_
            self.data_val = self.valid_dataset_
        if stage == "test" or stage == "predict":
            self.data_test = self.test_dataset_

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
