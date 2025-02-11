from torchmetrics import MeanSquaredError
class MeanSquaredErrorFlat(MeanSquaredError):
    def __init__(self, squared = True, num_outputs = 1, **kwargs):
        super().__init__(squared, num_outputs, **kwargs)
    def update(self, preds, target):
        preds = preds.reshape(-1)
        target = target.reshape(-1)
        return super().update(preds, target)