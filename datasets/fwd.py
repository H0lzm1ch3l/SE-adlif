import math
import os
from pathlib import Path
from typing import Any, Callable, Tuple, Union
import hydra
import numpy as np
import tonic
import torch
from tonic.transforms import Optional, ToFrame
from torchvision.datasets import DatasetFolder
from datasets.utils.pad_tensors import PadTensors
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as T


class EventDatasetFolder(DatasetFolder):
    
    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        ignore_first_timesteps = 0
    ) -> None:
        super().__init__(root=root, loader=loader, extensions=extensions, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file, allow_empty=allow_empty)
        self.ignore_first_timesteps = ignore_first_timesteps

    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        target = torch.tensor(target, dtype=torch.int64)
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        block_idx[:self.ignore_first_timesteps] = 0
        return events, target, block_idx
    
    
class FWDLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        spatial_factor: float = 0.15,
        time_factor: float = 1,
        time_window=5000,
        batch_size: int = 256,
        num_workers: int = 1,
        pad_to_min_size: int = 10,
        num_classes: int = 3,
        ignore_first_timesteps: int = 1,
    ) -> None:
        super().__init__()
        if not os.path.isabs(data_path):
            cwd = hydra.utils.get_original_cwd()
            data_path = os.path.abspath(os.path.join(cwd, data_path))
        self.data_path = os.path.join(data_path, "FWD")
        self.time_window = time_window
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = PadTensors()
        self.min_len = pad_to_min_size
        self.ignore_first_timesteps = ignore_first_timesteps  
        self.spatial_factor = spatial_factor
        self.time_factor = time_factor
        
        sensor_size = (320, 320, 2)
        sensor_size = (
            int(math.ceil(sensor_size[0] * self.spatial_factor)),
            int(math.ceil(sensor_size[1]) * self.spatial_factor),
            sensor_size[2],
        )
        self.sensor_size = sensor_size
        
        self.output_size = num_classes
        self.class_weights = torch.ones(
            size=(self.output_size,),
        )
        
        _event_to_tensor = ToFrame(
            sensor_size=self.sensor_size, time_window=self.time_window
        )
        event_to_tensor = lambda x: torch.from_numpy(_event_to_tensor(x)).float()
        
        if self.min_len > 0:
            def pad_to_min_len(x):
                if x.shape[0] < self.min_len:
                    pad = torch.zeros((self.min_len - x.shape[0], *self.sensor_size[::-1]))
                    x = torch.cat((x, pad), dim=0)
                return x
        else:
            def pad_to_min_len(x):
                return x
        
        # Data Agumentation Transforms
        transform_list = [
            tonic.transforms.Downsample(spatial_factor=spatial_factor, time_factor=time_factor),
            event_to_tensor,
            pad_to_min_len,
            # T.RandomResizedCrop(size=(48, 48), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # Flatten(),
        ]
        self.static_data_transform = tonic.transforms.Compose(transform_list)


    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        dataset = EventDatasetFolder(
            ignore_first_timesteps=self.ignore_first_timesteps,
            root=self.data_path,
            loader=np.load,
            extensions=".npy",
            transform=self.static_data_transform
        )
        dataset_indices = list(range(len(dataset)))
        dataset_targets = [dataset[i][1] for i in dataset_indices]
        # do train val test split
        train_indices, test_indices = train_test_split(dataset_indices, test_size=100, random_state=42, stratify=dataset_targets)
        train_indices, val_indices = train_test_split(train_indices, test_size=100, random_state=42, stratify=[dataset_targets[i] for i in train_indices])   
        self.data_train = torch.utils.data.Subset(dataset, train_indices)
        self.data_val = torch.utils.data.Subset(dataset, val_indices)
        self.data_test = torch.utils.data.Subset(dataset, test_indices) 

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        
    def test_dataloader(self):  
        return torch.utils.data.DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        