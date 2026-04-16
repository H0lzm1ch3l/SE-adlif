import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig
import math

R_m = 1.0

def gaussian(x, mu=0.0, sigma=0.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(
        2 * torch.tensor(math.pi)
    ) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        hight = 0.15
        lens = 0.5
        if True:
            temp = gaussian(input, mu=0.0, sigma=lens) * (1.0 + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        return grad_input * temp.float() * 0.5


act_fun_adp = ActFun_adp.apply


def mem_update_pra(inputs, mem, spike, v_th, tau_m, dt=1, device=None):
    alpha = torch.sigmoid(tau_m)
    mem = mem * alpha + (1 - alpha) * R_m * inputs - v_th * spike
    inputs_ = mem - v_th
    spike = act_fun_adp(inputs_)
    return mem, spike


def output_Neuron_pra(inputs, mem, tau_m, dt=1, device=None):
    alpha = torch.sigmoid(tau_m).to(device)
    mem = mem * alpha + (1 - alpha) * inputs
    return mem


class DHSSNNLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_size
        self.output_dim = cfg.get('n_neurons', cfg.get('output_dim'))
        if self.output_dim is None:
            raise ValueError("DHSNN layer requires n_neurons or output_dim")

        self.vth = cfg.get('vth', 0.5)
        self.dt = cfg.get('dt', 1)
        self.branch = cfg.get('branch', 4)
        self.test_sparsity = cfg.get('test_sparsity', False)
        self.sparsity = cfg.get('sparsity', 0.5)
        self.mask_share = cfg.get('mask_share', 1)

        self.pad = ((self.input_dim) // self.branch * self.branch + self.branch - self.input_dim) % self.branch
        self.dense = nn.Linear(self.input_dim + self.pad, self.output_dim * self.branch, bias=True)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, self.branch))
        self.register_buffer(
            'mask', torch.zeros(self.output_dim * self.branch, self.input_dim + self.pad)
        )
        self.create_mask()

        if cfg.get('tau_minitializer', 'uniform') == 'uniform':
            nn.init.uniform_(self.tau_m, cfg.get('low_m', 0), cfg.get('high_m', 4))
        else:
            nn.init.constant_(self.tau_m, cfg.get('low_m', 0))

        if cfg.get('tau_ninitializer', 'uniform') == 'uniform':
            nn.init.uniform_(self.tau_n, cfg.get('low_n', 0), cfg.get('high_n', 4))
        else:
            nn.init.constant_(self.tau_n, cfg.get('low_n', 0))

    def create_mask(self):
        input_size = self.input_dim + self.pad
        self.mask.zero_()
        for i in range(self.output_dim // self.mask_share):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                if self.test_sparsity:
                    start = j * input_size // self.branch
                    end = start + int(input_size * self.sparsity)
                    if end <= input_size:
                        indices = seq[start:end]
                    else:
                        indices = torch.cat(
                            [seq[start:], seq[: end - input_size]], dim=0
                        )
                else:
                    indices = seq[j * input_size // self.branch : (j + 1) * input_size // self.branch]
                for k in range(self.mask_share):
                    idx = (i * self.mask_share + k) * self.branch + j
                    self.mask[idx, indices] = 1.0

    def apply_mask(self):
        self.dense.weight.data = self.dense.weight.data * self.mask

    def initial_state(self, batch_size, device):
        self.mem = torch.rand(batch_size, self.output_dim, device=device)
        self.spike = torch.rand(batch_size, self.output_dim, device=device)
        if self.branch == 1:
            self.d_input = torch.rand(batch_size, self.output_dim, self.branch, device=device)
        else:
            self.d_input = torch.zeros(batch_size, self.output_dim, self.branch, device=device)
        self.v_th = torch.ones(batch_size, self.output_dim, device=device) * self.vth
        return None

    def forward(self, input_spike, state=None):
        padding = torch.zeros(
            input_spike.size(0), self.pad, device=input_spike.device, dtype=input_spike.dtype
        )
        k_input = torch.cat((input_spike.float(), padding), dim=1)
        beta = torch.sigmoid(self.tau_n)
        self.d_input = beta * self.d_input + (1 - beta) * self.dense(k_input).reshape(
            -1, self.output_dim, self.branch
        )
        l_input = self.d_input.sum(dim=2)
        self.mem, self.spike = mem_update_pra(
            l_input,
            self.mem,
            self.spike,
            self.v_th,
            self.tau_m,
            self.dt,
            device=input_spike.device,
        )
        return self.spike, state


class DHReadoutLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_size
        self.output_dim = cfg.get('n_neurons', cfg.get('output_dim'))
        if self.output_dim is None:
            raise ValueError("DHSNN readout requires n_neurons or output_dim")
        self.dt = cfg.get('dt', 1)
        self.dense = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        if cfg.get('tau_minitializer', 'uniform') == 'uniform':
            nn.init.uniform_(self.tau_m, cfg.get('low_m', 0), cfg.get('high_m', 4))
        else:
            nn.init.constant_(self.tau_m, cfg.get('low_m', 0))

    def initial_state(self, batch_size, device):
        self.mem = torch.rand(batch_size, self.output_dim, device=device)
        return None

    def forward(self, input_spike, state=None):
        d_input = self.dense(input_spike.float())
        self.mem = output_Neuron_pra(
            d_input,
            self.mem,
            self.tau_m,
            self.dt,
            device=input_spike.device,
        )
        return self.mem, state


class DHSNN(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.ignore_target_idx = -1
        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.batch_size = cfg.dataset.batch_size
        self.dropout = cfg.get('dropout', 0.0)

        self.lr = cfg.lr
        self.factor = cfg.factor
        self.patience = cfg.patience
        self.auto_regression = cfg.get('auto_regression', False)

        self.l1 = DHSSNNLayer(cfg.l1)
        if cfg.two_layers:
            self.l2 = DHSSNNLayer(cfg.l2)
        self.out_layer = DHReadoutLayer(cfg.l_out)

        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        device = inputs.device
        self.l1.initial_state(batch_size, device)
        if hasattr(self, 'l2'):
            self.l2.initial_state(batch_size, device)
        self.out_layer.initial_state(batch_size, device)

        out_sequence = torch.zeros(
            (batch_size, inputs.shape[1], self.output_size), device=device
        )
        sparsity_sequences = torch.zeros(
            (inputs.shape[1], 2 if hasattr(self, 'l2') else 1), device=device
        )
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))

        for t, x_t in enumerate(inputs.unbind(1)):
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()
            out, _ = self.l1(x_t, None)
            sparsity_sequences[t, 0] = out.mean()
            if hasattr(self, 'l2'):
                out, _ = self.l2(out, None)
                sparsity_sequences[t, 1] = out.mean()
            out, _ = self.out_layer(out, None)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out_sequence[:, t] = out

        self.sparsity_sequences = sparsity_sequences.mean(dim=0)
        return out_sequence

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.l1.apply_mask()
        if hasattr(self, 'l2'):
            self.l2.apply_mask()

    def process_predictions_and_compute_losses(self, outputs, targets, block_idx):
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
        if self.auto_regression:
            single_step_prediction_limit = int(math.ceil(0.5 * outputs.shape[1]))
            outputs = outputs[:, single_step_prediction_limit:].squeeze()
            targets = targets[:, single_step_prediction_limit + 1 :].squeeze()
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
        outputs = self(inputs)
        outputs_reduce, loss, block_idx = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx
        )
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
        outputs_reduce, loss, block_idx = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx
        )
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
        outputs_reduce, loss, block_idx = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx
        )
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
