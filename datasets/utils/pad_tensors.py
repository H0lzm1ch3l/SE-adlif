import torch
from torch.nn.utils.rnn import pad_sequence

class PadTensors:
    """This class implement the specific collate logic for our data
    each call function are resolve at object creation
    this prevent control flow inside the dataset loop 
    """
    
    def __init__(self, batch_first: bool = True, require_padding: bool = True):
        self.batch_first = batch_first
        self.batch_dim = 0 if batch_first else 1
        self.temp_dim = 1 if batch_first else 0
        self.require_padding = require_padding
        self.call_fn = self._call_with_padding if self.require_padding else self._call
    @staticmethod
    def _call(batch, batch_first: bool):
        res = list(zip(*batch))  # type: ignore
        res = [PadTensors.collate_tensor_fn(x) for x in res]
        inputs, target, block_idx = res
        dummy_target = torch.full_like(target[:, 0], fill_value=-1).unsqueeze(1)
        target = torch.concatenate((dummy_target, target), dim=1)
        return inputs, target, block_idx
    @staticmethod
    def _call_with_padding(batch, batch_first: bool):        
        inputs, target_list, block_idx = list(zip(*batch))  # type: ignore

        # If target is a scalar, convert it to a tensor
        if len(target_list[0].shape) == 0:
            target_list = torch.tensor(target_list).unsqueeze(1)

        target = pad_sequence(
        target_list, batch_first=batch_first, padding_value=-1) 

        # Padding block (zero-valued time steps of the block_idx) MUST have target of -1 !!
        # target = torch.concatenate((torch.full((target.shape[0], 1), fill_value=-1), target), dim=1)
        dummy_target = torch.full_like(target[:, 0], fill_value=-1).unsqueeze(1)
        
        target = torch.concatenate((dummy_target, target), dim=1)
        inputs = pad_sequence(inputs, batch_first=batch_first, padding_value=0)

        block_idx = pad_sequence(
            block_idx, batch_first=batch_first, padding_value=0
            ).long()
        return inputs, target, block_idx

    def __call__(self, batch):
        return self.call_fn(batch=batch, batch_first=self.batch_first)
    @staticmethod
    def collate_tensor_fn(batch):
        elem = batch[0]
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)