import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Sequence, Union

import numpy as np
import torch

@dataclass
class DiskCachedDataset(torch.utils.data.Dataset):
    dataset: Sequence
    cache_path: Union[str, Path]
    num_copies: int = 1
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    full_transform: Optional[Callable] = None
    
    def __post_init__(self):
        super().__init__()
        self.cache_path = Path(self.cache_path)
        self.cache_path.mkdir(exist_ok=True, parents=True)
        if self.dataset is None:
            filenames = [
                name for name in self.cache_path.iterdir() if name.is_file()
                ]
            self.n_samples = (len(filenames) // self.num_copies)
        else:
            self.n_samples = len(self.dataset)
        
    def io(self, key, item=None):
        data = None
        if item is None:
            item = key
        try:
            f = open(self.cache_path / key, 'rb')
            buffer = io.BytesIO(f.read())
            data = torch.load(buffer)
        except Exception as exc:
            if not isinstance(exc, FileNotFoundError):
                print("Excpection (not FileNotFoundError) occured while trying to load from cache: {}".format(exc))
            data = self.dataset[item]
            try:
                torch.save(data, self.cache_path / key, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                print("Excpection occured while trying to save to cache: {}".format(exc))
        return data
    
    context_transform: Optional[Callable] = None

    def __getitem__(self, item) -> Union[Tuple[object, object, object],
                                         Tuple[object, object]]:
        copy = np.random.randint(self.num_copies)
        key = f"{item}_{copy}"
        
        raw_data = self.io(key, item)
        data, targets, *meta = raw_data
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        if self.full_transform is not None:
            data, targets, *meta = self.full_transform(data, targets, *meta)
        return data, targets, *meta

    def __len__(self):
        return self.n_samples