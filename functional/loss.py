import math
from typing import Optional, Union
import torch
from torchaudio.transforms import MelSpectrogram
reduce_map = {
    'mean': lambda x: torch.mean(x),
    'sum': lambda x: torch.sum(x),
    'none': lambda x: x,
}
class MultiScaleMelSpetroLoss(torch.nn.Module):
    def __init__(self, sampling_rate, n_mels, min_windows_power, max_windows_power, reduce='mean'):
        super().__init__()
        self.windows_size = [2**i for i in range(min_windows_power, max_windows_power+1)]
        self.register_buffer("log_scale", torch.tensor([math.sqrt(s/2) for s in self.windows_size]))
        # I'm not quite sure what is the correct parameter for n_fft, It's imply to be s (in soundstream ) but it should be higher than n_mels
        # but it's not clear by how much
        self.min_n_fft = 8*n_mels
        self.mel_banks = torch.nn.ModuleList([MelSpectrogram(sampling_rate, max(s, self.min_n_fft), s, s//4, n_mels=n_mels, f_min=sampling_rate/s, f_max=sampling_rate/2, normalized=True) for s in self.windows_size])
        self.eps: float = 1e-9
        self.reduce = reduce
        self.reduce_fn = reduce_map[self.reduce]
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        x = [mb(inputs.squeeze()) for mb in self.mel_banks]
        y = [mb(targets.squeeze()) for mb in self.mel_banks]
        log_x = [torch.log(elem+self.eps) for elem in x]
        log_y = [torch.log(elem+self.eps) for elem in y]
        # I have no idea that is the correct implentation none of the paper code follows their equation 
        # implementation 1: direct translation of https://github.com/google-research/google-research/blob/master/ged_tts/distance_function/spectral_ops.py
        # average over 1/(n_mels*n_time) for linear and 1/n_mels *1/(n_times)**0.5 for log_delta 
        # (it make somewhat sense to do the square root of the mean than the mean of the square root)
        linear = torch.stack([torch.abs(elem_x - elem_y).mean(-1).mean(-1) for elem_x, elem_y in zip(x,y)], dim=-1).sum(dim=-1)
        log_delta = torch.stack([
            scale*torch.sqrt(
                torch.square(elem_x - elem_y).mean(-1) + self.eps
                ).mean(-1) for scale, elem_x, elem_y in zip(self.log_scale,log_x, log_y)
            ], dim=-1).sum(dim=-1)
        # implementation 2: from https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/soundstream.py a popular audio ml project
        # here the sum is done over the n_mels and mean over times
        
        # linear = torch.stack([torch.abs(elem_x - elem_y).sum(-2).mean(-1) for elem_x, elem_y in zip(x,y)], dim=-1).sum(dim=-1)
        # log_delta = torch.stack([
        #             scale*torch.sqrt(
        #                 torch.square(elem_x - elem_y).sum(-2) + self.eps
        #                 ).mean(-1) for scale, elem_x, elem_y in zip(self.log_scale,log_x, log_y)
        #             ], dim=-1).sum(dim=-1)
        loss = linear + log_delta
        return self.reduce_fn(loss)
        
def get_per_layer_spike_probs(states_list: list[torch.Tensor], 
                              block_idx):
    """
    Iterate over the recorded states of each layers,
    if this states correspond on to a spiking neurons states.
    Retrieved average spike probability for this layer for regularization purposes.
    
    """
    spike_probs =  [
            get_spike_prob(states[1], block_idx)
            for states in states_list if states.size(0) > 1
        ]
    return spike_probs


def get_spike_prob(z, block_idx):
    """
    Determined the average spike probability for each neuron of a specific layer.
    The spike probability is sum(z^t)/T, where T is the sample duration
    determined as the sum of all valid/non-padded time-steps.
    
    """
    z = z[:, 1:]
    
    # create a tensor that have the same shape as z except for the temporal dimension
    # the temporal dimension correspond to result of the scatter operation
    # for the padded timesteps (spike_proba_per_block[:, 0]) and non-padded timesteps
    # (spike_proba_per_block[:, 1]) only the former is of interest to us for the regularization loss
     
    spike_proba_per_block = torch.zeros(
        size=(z.shape[0], 2, z.shape[2]), device=z.device
    )
    # determined all non-padded time-steps
    # assuming that padded time-steps are always summed to block_idx[:, 0]
    padded_timesteps_mask = (block_idx != torch.tensor(0)).long()
    
    spike_proba_per_block.scatter_reduce_(
        1,
        # all padded time-steps are averaged to the 0-index
        # all valid time-steps are averaged to the 1-index
        padded_timesteps_mask.broadcast_to(z.shape),
        z,
        reduce="mean",
        include_self=False,
    )
    # mean over all batches for the non-padded timesteps
    return spike_proba_per_block[:, 1].mean(dim=0) 


def snn_regularization(spike_proba_per_layer: list[torch.Tensor], target: Union[float | list[float]], 
                       layer_weights: torch.Tensor,
                       reg_type: str, 
                       reduce_neuron: str = "mean",
                       reduce_layer: str = "mean",):
    """
    Determined the regularization loss 

    Args:
        spike_proba_per_layer (torch.Tensor): spike probability for each neuron per layer
        spike_proba_par_layer[i] is a (nums_neuron, ) tensor representing the spike probability 
        for each neuron of layer i 
        target (float): the spike probability target
        reg_type (str): the regularization type
    
    upper regularization: only regularized neurons where the spike probability is higher than the target
    lower regularization: only regularized neurons where the spike probability is lower than the target 
    both: regularized neuron with respect the squared distance of neuron spike probability and the target

    Raises:
        NotImplementedError: Raise error if reg_type is unknown 

    Returns:
        torch.Tensor: the regularization loss averaged over the total numbers of neurons 
    """
    if isinstance(target, float):
        target = [target,] * len(spike_proba_per_layer)
    if reg_type == "lower":
        reg = [(torch.relu(tg - s) ** 2) for tg, s in zip(target, spike_proba_per_layer)]
    elif reg_type == "upper":
        reg = [(torch.relu(s - tg) ** 2) for tg, s in zip(target, spike_proba_per_layer)]
    elif reg_type == "both":
        reg = [((s - tg) ** 2) for tg, s in zip(target, spike_proba_per_layer)]
    else:
        raise NotImplementedError(
            f"Regularization type: {reg_type} is not implemented, valid type are [upper, lower, both]"
        )
    if reduce_neuron == 'mean':
        reg = torch.stack([r.mean() for r in reg])
    elif reduce_neuron == 'sum':
        reg = torch.stack([r.sum() for r in reg])
    res = (layer_weights * reg)
    log_dict = {f"{reg_type}_layer_{k}": res[k] for k in range(len(res))}
    if reduce_layer == 'mean':
        return res.mean(), log_dict
    elif reduce_layer == 'sum':
        return res.sum(), log_dict
    
def calculate_weight_decay_loss(model):
    weight_decay_loss = 0
    for name, param in model.named_parameters():
        if (
            "bias" not in name
            and "tau" not in name
            and "beta" not in name
            and "coeff" not in name
        ):
            weight_decay_loss += torch.mean(param**2)
    return weight_decay_loss
