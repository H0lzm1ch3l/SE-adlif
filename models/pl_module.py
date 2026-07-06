import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig

from models.alif import EFAdLIF, SEAdLIF
from models.alif2 import AdLIF2
from models.li import LI
from models.lif import LIF
from models.rnn import LSTMCellWrapper
from models.mclif import MCLIF
from models.mcalif import MCAdLIF
from models.mclif_v2 import MCLIF2
from models.resnet import SpikingResNet


layer_map = {
    "lif": LIF,
    "mclif": MCLIF,
    "mclif_v2": MCLIF2,
    "mcalif": MCAdLIF,
    "se_adlif": SEAdLIF,
    "ef_adlif": EFAdLIF,
    "adlif2": AdLIF2,
    'lstm': LSTMCellWrapper,
}


def normalize_hidden_layers(cfg):
    if cfg.get('hidden_layers') is not None:
        return cfg.hidden_layers
    hidden_layers = [cfg.l1]
    if cfg.get('two_layers', False):
        hidden_layers.append(cfg.l2)
    return hidden_layers


class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.ignore_target_idx = -1
        self.two_layers = cfg.two_layers
        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.batch_size = cfg.dataset.batch_size
        self.dropout = cfg.dropout

        # For learning rate scheduling (used for oscillation task)
        self.lr = cfg.lr
        self.factor = cfg.factor
        self.patience = cfg.patience

        self.auto_regression = cfg.get('auto_regression', False)

        # Define the model hidden layers
        self.hidden_layers_cfg = normalize_hidden_layers(cfg)
        self.hidden_layers = nn.ModuleList(
            [layer_map[layer_cfg.cell](layer_cfg) for layer_cfg in self.hidden_layers_cfg]
        )
        self.num_hidden_layers = len(self.hidden_layers)
        self.two_layers = cfg.get('two_layers', self.num_hidden_layers > 1)
        self.out_layer = LI(cfg.l_out)
        
        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()

    def forward(
        self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # print(f"Input shape: {inputs.shape}")

        states = [layer.initial_state(inputs.shape[0], inputs.device) for layer in self.hidden_layers]
        s_out = self.out_layer.initial_state(inputs.shape[0], inputs.device)

        out_sequence = torch.zeros((inputs.shape[0], inputs.shape[1], self.out_layer.out_features), device=inputs.device)
        sparsity_sequences = torch.zeros((inputs.shape[1], self.num_hidden_layers), device=inputs.device)
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))

        # Iterate over each time step in the data
        for t, x_t in enumerate(inputs.unbind(1)):
            
            # Auto-regression for oscillator task
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()

            out = x_t
            for layer_idx, layer in enumerate(self.hidden_layers):
                out, states[layer_idx] = layer(out, states[layer_idx])
                sparsity_sequences[t, layer_idx] = out.mean()
                out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)

            out, s_out = self.out_layer(out, s_out)
            out_sequence[:, t] = out
        
        self.sparsity_sequences = sparsity_sequences.mean(dim=0)
        return out_sequence

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        for layer in self.hidden_layers:
            if hasattr(layer, 'apply_parameter_constraints'):
                layer.apply_parameter_constraints()
        self.out_layer.apply_parameter_constraints()

    def process_predictions_and_compute_losses(self, outputs, targets, block_idx):
        """
        Process the model output into prediction
        with respect to the temporal segmentation defined by the
        block_idx tensor.
        Then compute losses
        Args:
            outputs (torch.Tensor): full outputs
            targets (torch.Tensor): targets
            block_idx (torch.Tensor): tensor of index that determined which temporal segements of
            output time-step depends on which specific target,
            used by the scatter reduce operation.

        Returns:
            (): _description_
        """
        # compute softmax for every time-steps with respect to
        # the number of class
        if self.auto_regression:
            targets = targets[:, 1:]
            l2_loss = (outputs - targets) ** 2
            
            block_outputs = torch.zeros(
                size=(targets.shape[0], 2, outputs.shape[2]),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            _block_idx = block_idx.unsqueeze(2).expand(size=(-1, -1, outputs.size(2)))
            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=_block_idx,
                src=l2_loss,
                reduce="mean",
                include_self=False,
            )
            block_output = block_output[:, 1]
            outputs_reduce = outputs
            loss = block_output.mean()
        else:
            if self.output_func == "softmax":
                outputs = torch.softmax(outputs, -1)
                reduction = "sum"
            else:
                reduction = "mean"
            # create a zero array of size (batch, number_of_targets, number_of_classes)
            # this will be used to defined the prediction for each targets for each classes
            block_outputs = torch.zeros(
                size=(targets.size(0), targets.size(1), outputs.size(2)),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            block_idx = block_idx.unsqueeze(-1)

            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=block_idx.broadcast_to(outputs.shape),
                src=outputs,
                reduce=reduction,
                include_self=False,
            )


            outputs_reduce = block_output.reshape(-1, outputs.size(-1))
            targets_reduce = targets.flatten()

            block_mask = torch.where(targets_reduce != self.ignore_target_idx)

            loss = self.loss(outputs_reduce[block_mask].float(), targets_reduce[block_mask])
        return (outputs_reduce, loss, block_idx)

    def update_and_log_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        metrics: torchmetrics.MetricCollection,
        prefix: str,
    ):
        """
        Method centralizing the metrics logging mecanisms.

        Args:
            outputs_reduce (torch.Tensor): output prediction
            targets_reduce (torch.Tensor): target
            loss (float): loss
            metrics (torchmetrics.MetricCollection): collection of torchmetrics metrics
            aux_metrics (dict): auxiliary metrics that do not
            fit the torchmetrics logic
            prefix (str): prefix defining the stage of model either
            "train_": training stage
            "val_": validation stage
            "test_": testing stage
            Those prefix prevent clash of names in the logger.

        """
        if self.auto_regression:
            single_step_prediction_limit = int(math.ceil(0.5*outputs.shape[1]))
            outputs = outputs[:, single_step_prediction_limit:].squeeze()
            targets = targets[:, single_step_prediction_limit+1:].squeeze()
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
        else:
            targets = targets.flatten()

        metrics(outputs, targets)
        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )
        
        if hasattr(self, 'sparsity_sequences'):
            for i, sparsity in enumerate(self.sparsity_sequences):
                self.log(
                    f"{prefix}sparsity_layer_{i+1}",
                    sparsity,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=True if prefix == "train_" else False,
                )

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(
            inputs,
        )
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.train_metric,
            prefix="train_",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.val_metric,
            prefix="val_",
        )

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.test_metric,
            prefix="test_",
        )

        return loss

    def init_metrics_and_loss(self):
        if self.auto_regression:
            metrics = torchmetrics.MetricCollection(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                }
            )
            self.loss = MSELoss()
        else:
            metrics = torchmetrics.MetricCollection(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass",  # type: ignore
                        num_classes=self.output_size,
                        average="micro",
                        ignore_index=self.ignore_target_idx,
                    )
                }
            )
            self.loss = CrossEntropyLoss(ignore_index=self.ignore_target_idx)
        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.tracking_mode,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.tracking_metric,
            },
        }


class SpikingResNetSNN(pl.LightningModule):
    """
    PyTorch Lightning module for Spiking ResNets with configurable neuron models.
    Supports multiple stages of residual blocks with different depths and widths.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.ignore_target_idx = -1
        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.batch_size = cfg.dataset.batch_size
        self.dropout = cfg.get('dropout', 0.0)

        # For learning rate scheduling
        self.lr = cfg.lr
        self.factor = cfg.factor
        self.patience = cfg.patience

        self.auto_regression = cfg.get('auto_regression', False)

        # Define the ResNet model
        self.model = SpikingResNet(cfg)
        
        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the ResNet.
        
        Args:
            inputs: (batch_size, time_steps, input_size)
        
        Returns:
            outputs: (batch_size, time_steps, num_classes)
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        states = self.model.initial_state(batch_size, device)
        out_sequence = []
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))

        for t, x_t in enumerate(inputs.unbind(1)):
            # Auto-regression for oscillator task
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()

            out, states = self.model(x_t, states)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out_sequence.append(out)
        
        return torch.stack(out_sequence, dim=1)

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.model.apply_parameter_constraints()

    def process_predictions_and_compute_losses(self, outputs, targets, block_idx):
        """
        Process the model output into prediction with respect to the temporal segmentation.
        Then compute losses.
        """
        if self.auto_regression:
            targets = targets[:, 1:]
            l2_loss = (outputs - targets) ** 2
            
            block_outputs = torch.zeros(
                size=(targets.shape[0], 2, outputs.shape[2]),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            _block_idx = block_idx.unsqueeze(2).expand(size=(-1, -1, outputs.size(2)))
            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=_block_idx,
                src=l2_loss,
                reduce="mean",
                include_self=False,
            )
            block_output = block_output[:, 1]
            outputs_reduce = outputs
            loss = block_output.mean()
        else:
            if self.output_func == "softmax":
                outputs = torch.softmax(outputs, -1)
                reduction = "sum"
            else:
                reduction = "mean"
            
            block_outputs = torch.zeros(
                size=(targets.size(0), targets.size(1), outputs.size(2)),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            block_idx = block_idx.unsqueeze(-1)

            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=block_idx.broadcast_to(outputs.shape),
                src=outputs,
                reduce=reduction,
                include_self=False,
            )

            outputs_reduce = block_output.reshape(-1, outputs.size(-1))
            targets_reduce = targets.flatten()

            block_mask = torch.where(targets_reduce != self.ignore_target_idx)

            loss = self.loss(outputs_reduce[block_mask].float(), targets_reduce[block_mask])
        return (outputs_reduce, loss, block_idx)

    def update_and_log_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        metrics: torchmetrics.MetricCollection,
        prefix: str,
    ):
        """Method centralizing the metrics logging mechanisms."""
        if self.auto_regression:
            single_step_prediction_limit = int(math.ceil(0.5 * outputs.shape[1]))
            outputs = outputs[:, single_step_prediction_limit:].squeeze()
            targets = targets[:, single_step_prediction_limit + 1:].squeeze()
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
        else:
            targets = targets.flatten()

        metrics(outputs, targets)
        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.train_metric,
            prefix="train_",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.val_metric,
            prefix="val_",
        )

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.test_metric,
            prefix="test_",
        )

        return loss

    def init_metrics_and_loss(self):
        if self.auto_regression:
            metrics = torchmetrics.MetricCollection(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                }
            )
            self.loss = MSELoss()
        else:
            metrics = torchmetrics.MetricCollection(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass",
                        num_classes=self.output_size,
                        average="micro",
                        ignore_index=self.ignore_target_idx,
                    )
                }
            )
            self.loss = CrossEntropyLoss(ignore_index=self.ignore_target_idx)
        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.tracking_mode,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.tracking_metric,
            },
        }