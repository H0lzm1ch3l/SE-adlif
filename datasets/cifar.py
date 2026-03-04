import math
import os

import hydra
import tonic
from tonic.datasets import CIFAR10DVS
import torch
from tonic.transforms import Optional, ToFrame
from datasets.utils.pad_tensors import PadTensors
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from datasets.utils.transforms import Flatten


class CIFAR10DVSWrapper(CIFAR10DVS):
    
    def __init__(
        self,
        save_to: str,
        transform = None,
        target_transform = None,
        transforms = None,
        ignore_first_timesteps: int = 10,
    ):
        super().__init__(save_to, transform, target_transform, transforms)
        self.ignore_first_timesteps = ignore_first_timesteps
    
    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        block_idx[:self.ignore_first_timesteps] = 0
        return events, target, block_idx
    

class CIFAR10DVSLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        spatial_factor: float = 0.375,
        time_factor: float = 1e-3,
        window_size: float = 129.0,
        batch_size: int = 256,
        num_workers: int = 1,
        pad_to_min_size: int = 300,
        num_classes: int = 10,
        ignore_first_timesteps: int = 10,
    ) -> None:
        super().__init__()
        if not os.path.isabs(data_path):
            cwd = hydra.utils.get_original_cwd()
            data_path = os.path.abspath(os.path.join(cwd, data_path))
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = PadTensors()
        self.min_len = pad_to_min_size
        self.ignore_first_timesteps = ignore_first_timesteps  
        self.spatial_factor = spatial_factor
        self.time_factor = time_factor
        
        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
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
            sensor_size=self.sensor_size, time_window=self.window_size
        )
        event_to_tensor = lambda x: torch.from_numpy(_event_to_tensor(x)).float()
        
        if self.min_len > 0:
            def pad_to_min_len(x):
                if x.shape[0] < self.min_len:
                    pad = torch.zeros((self.min_len - x.shape[0], 1, x.shape[-1]))
                    x = torch.cat((x, pad), dim=0)
                return x
        else:
            def pad_to_min_len(x):
                return x
        transform_list = [
            tonic.transforms.Downsample(spatial_factor=spatial_factor, time_factor=time_factor),
            event_to_tensor,
            pad_to_min_len,
            Flatten(),
        ]
        self.static_data_transform = tonic.transforms.Compose(transform_list)

    def prepare_data():
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        dataset_indices = list(range(len(CIFAR10DVSWrapper(save_to=self.data_path))))
        dataset_targets = [CIFAR10DVSWrapper(save_to=self.data_path)[i][1] for i in dataset_indices]
        print(f"Dataset indices: {dataset_indices.shape}")
        exit()
        # do train val test split
        train_indices, test_indices = train_test_split(dataset_indices, test_size=10000, random_state=42, stratify=dataset_targets)
        train_indices, val_indices = train_test_split(train_indices, test_size=10000, random_state=42, stratify=[dataset_targets[i] for i in train_indices])   
        self.data_train = torch.utils.data.Subset(CIFAR10DVSWrapper(save_to=self.data_path, transform=self.static_data_transform, ignore_first_timesteps=self.ignore_first_timesteps), train_indices)
        self.data_val = torch.utils.data.Subset(CIFAR10DVSWrapper(save_to=self.data_path, transform=self.static_data_transform, ignore_first_timesteps=self.ignore_first_timesteps), val_indices)
        self.data_test = torch.utils.data.Subset(CIFAR10DVSWrapper(save_to=self.data_path, transform=self.static_data_transform, ignore_first_timesteps=self.ignore_first_timesteps), test_indices)
    
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
        
                
        
        
    
        
