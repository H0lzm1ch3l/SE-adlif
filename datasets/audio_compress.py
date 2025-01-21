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
import tarfile
import urllib.request

def download_and_extract_tar_gz(url, extract_to):
    """
    Downloads a .tar.gz file from the specified URL and extracts it to the given directory.

    :param url: str - The URL to the .tar.gz file.
    :param extract_to: str - The path to the directory where the archive will be extracted.
    """
    extract_to_path = Path(extract_to)

    # Ensure the destination directory exists
    extract_to_path.mkdir(parents=True, exist_ok=True)

    # Download the file
    tar_gz_path = extract_to_path / Path(url).name
    print(f"Downloading {url} to {tar_gz_path}...")

    with urllib.request.urlopen(url) as response, open(tar_gz_path, 'wb') as out_file:
        file_size = int(response.getheader('Content-Length', 0))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=tar_gz_path.name) as pbar:
            while chunk := response.read(1024):
                out_file.write(chunk)
                pbar.update(len(chunk))

    print("Download complete.")

    # Extract the file
    print(f"Extracting {tar_gz_path} to {extract_to}")
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_to_path)
    print("Extraction complete.")

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


def process_resampling_and_normalize(wav_file, output_path, target_sample_rate, norm_func):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(wav_file)

    with torch.no_grad():
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate,
            resampling_method='sinc_interp_kaiser'
        )
        waveform = norm_func(waveform)
    # Define the output file path (same name as the original)
    output_file = output_path / wav_file.name

    # Save the resampled audio to the output path
    torchaudio.save(output_file, waveform, target_sample_rate, encoding="PCM_S", bits_per_sample=16)


def resample_normalize_and_save_wav_files(input_path, output_path, base_freq, new_freq, norm_func):
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all .wav files to process
    wav_files = list(input_path.rglob("*.wav"))

    # Use multiprocessing to speed up the process
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_resampling_and_normalize, wav_file, output_path, new_freq, norm_func)
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

    # Determine the number of frames (samples) to load
    if chunk_size == -1:
        # If chunk_size == -1, load the entire file
        waveform, sample_rate = torchaudio.load(wav_file)
        return waveform

    # Use torchaudio.load to load only the chunk
    waveform, sample_rate = torchaudio.load(
        wav_file,
        frame_offset=start_sample,
        num_frames=chunk_size
    )
    return waveform



norm_map = {
    "none": lambda x: x,
    "-1_1": lambda x: -1 + 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x)),
    "peak": lambda x: x/torch.max(torch.abs(x)),
    "0_1": lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)),
    "standard": lambda x: (x - torch.mean(x)) / (torch.var(x) + 1e-9),
}


class LibriTTS(Dataset):
    def __init__(
        self,
        save_to: str,
        cache_path: str = None,
        sampling_freq: int = 24_000,
        sample_length: int = -1,
        normalization: str = "none",
        train: bool = True,
        debug: bool = True,
        transform=None,
        target_transform=None,
        full_transform=None,
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
        dev_clean_path = "https://openslr.elda.org/resources/60/dev-clean.tar.gz"
        dev_clean_md5 = "0c3076c1e5245bb3f0af7d82087ee207"
        test_clean_path = "https://openslr.elda.org/resources/60/test-clean.tar.gz"
        test_clean_md5 = "7bed3bdb047c4c197f1ad3bc412db59f"
        train_clean_path = "https://openslr.elda.org/resources/60/train-clean-100.tar.gz"
        train_clean_md5 = "4a8c202b78fe1bc0c47916a98f3a2ea8"
        save_to = Path(save_to)
        download_path = save_to / "LibriTTS"
        
        if debug:
             try_path = download_path / "dev-clean"
             download_path = dev_clean_path
        elif train:
            try_path = download_path / "train-clean"
            download_path = train_clean_path
        else:
            try_path = download_path / "test-clean"
            download_path = test_clean_path
            
        if not try_path.exists():
            download_and_extract_tar_gz(download_path, save_to)
            
        self.transform = transform
        self.target_transform = target_transform
        self.full_transform = full_transform
        self.get_metadata = get_metadata
        self.norm_type = normalization
        self.norm_func = norm_map[normalization]
        # self.root_dir = Path(download_root)
        self.cache_path = Path(cache_path) / "LibriTTS"
        split = "debug"
        if not debug:
            split = "train" if train else "test"
        full_split = split + "/24000"
        freq_split = split + f"/{sampling_freq}"
        freq_split += f"_{self.norm_type}"
        full_split_dir = self.cache_path / full_split
        freq_split_dir = self.cache_path / freq_split
        if (full_split_dir).exists() and any((full_split_dir).iterdir()):
            print("directory already exist")
        else:
            # for now only get debug
            copy_wave_files_with_path_names(try_path, full_split_dir)
        # 24_000/full split exist

        if not freq_split_dir.exists() or not any(freq_split_dir.iterdir()):
            # resample
            print(f"Resample wave file from 24kHz to {sampling_freq}Hz")
            resample_normalize_and_save_wav_files(
                full_split_dir, freq_split_dir, 24_000, sampling_freq, norm_func=self.norm_func
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

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        inputs: torch.Tensor = get_chunk_by_id(
            index, self.chunk_map, self.sample_length, self.norm_func
        ).T
        targets = inputs.clone()
        block_idx = torch.ones((inputs.shape[0],), dtype=torch.int32)
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.full_transform is not None:
            inputs, targets, block_idx = self.full_transform(inputs, targets, block_idx)
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
        debug: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fits_into_ram = fits_into_ram
        self.collate_fn = PadTensors(require_padding=False)
        self.normalization = normalization
        self.max_sample = max_sample
        
        if not os.path.isabs(data_path):
            cwd = hydra.utils.get_original_cwd()
            data_path = os.path.abspath(os.path.join(cwd, data_path))

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

        self.train_dataset_ = LibriTTS(
            save_to=data_path,
            cache_path=cache_path,
            sampling_freq=sampling_freq,
            sample_length=sample_length,
            normalization=normalization,
            debug=debug,
            train=True,
            transform=None,
            target_transform=None,
            full_transform=delay_transform,
        )
        # self.cache_path = (
        #     cache_path
        #     + f"/LibriTTS_hz-{sampling_freq}_sl-{sample_length}_norm-{normalization}"
        # )
        

        # def zero_inputs(inputs, targets, block_idx):
        #     rd = torch.rand(()).item()
        #     if rd < zero_input_proba:
        #         inputs = torch.zeros_like(inputs)
        #         targets = torch.zeros_like(targets)
        #     return inputs, targets, block_idx
        # def compose_full_transform(inputs, targets, block_idx):
        #     inputs, targets, block_idx = delay_transform(inputs, targets, block_idx)
        #     return zero_inputs(inputs, targets, block_idx)

        # self.train_dataset_ = DiskCachedDataset(
        #     self.train_dataset_,
        #     cache_path=self.cache_path,
        #     full_transform=compose_full_transform
        # )
        n_sample = len(self.train_dataset_)
        # always aim for 10*batch_size samples per valid and test set rest is train set
        # this prevents that too much data is used for validation when the dataset is very large
        i = 10
        while i > 1 and n_sample - 2*batch_size*i < 0:
            i -= 1
        if debug:
            self.train_dataset_, self.valid_dataset_, self.test_dataset_ = (
                torch.utils.data.random_split(
                    self.train_dataset_,
                    [n_sample - 2*i*batch_size, i*batch_size, i*batch_size],
                    generator=None,
                )
            )
        else:
            self.train_dataset_, self.valid_dataset_  = (
                torch.utils.data.random_split(
                    self.train_dataset_,
                    [n_sample - 2*i*batch_size, 2*i*batch_size,],
                    generator=None,
                )
            )
            self.test_dataset_ = LibriTTS(
                save_to=data_path,
                cache_path=cache_path,
                sampling_freq=sampling_freq,
                sample_length=sample_length,
                normalization=normalization,
                debug=debug,
                train=False,
                transform=None,
                target_transform=None,
                full_transform=delay_transform,
            )
        # create a sampler for self.train_dataset, that randomly sub-sample "self.max_sample" samples from
        # the total dataset length, this is not required if self.max_sample = -1 (total dataset)
        if self.max_sample != -1:
            self.train_sampler = torch.utils.data.RandomSampler(self.train_dataset_, replacement=False, num_samples=self.max_sample)
        else:
            self.train_sampler = None

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
            sampler=self.train_sampler,
            shuffle=False,
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
