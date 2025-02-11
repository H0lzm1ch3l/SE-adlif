import torch
from torchmetrics import MeanSquaredError
class MeanSquaredErrorFlat(MeanSquaredError):
    def __init__(self, squared = True, num_outputs = 1, **kwargs):
        super().__init__(squared, num_outputs, **kwargs)
    def update(self, preds, target):
        preds = preds.reshape(-1)
        target = target.reshape(-1)
        return super().update(preds, target)


def spike_probability_metric_per_layer(
    spike_probs_list: list[torch.Tensor]):
    """
    Return the mean spike probability per layer as a dictonary in order to visualized 
    them via the logger.

    Args:
        spike_probs_list (list[torch.Tensor]): the list of spike probability
            per neurons per layer
        prefix (str): The metric prefix.

    Returns:
        dict[str, torch.Tensor]: The by layer mean spike probability 
    """
    with torch.no_grad():
        z_dict = {
                    f"spike_prob_{i+1}": spike_probs.mean() for i, spike_probs in enumerate(spike_probs_list)
                    }
    return z_dict