import math
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig
from pytorch_lightning.utilities import grad_norm

from functional.loss import MultiScaleMelSpetroLoss, get_per_layer_spike_probs, snn_regularization
from models.alif import EFAdLIF, SEAdLIF
from models.li import LI
from models.lif import LIF
from models.rnn import LSTMCellWrapper
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)
torch._dynamo.config.cache_size_limit = 64
layer_map = {
    "lif": LIF,
    "se_adlif": SEAdLIF,
    "ef_adlif": EFAdLIF,
    'lstm': LSTMCellWrapper,
}
class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = layer_map[cfg.l1.cell](cfg.l1)
        self.l2 = layer_map[cfg.l2.cell](cfg.l2)
        self.l1_spike = torch.empty(size=())
        self.l2_spike = torch.empty(size=())
        self.dropout = cfg.dropout

    @torch.jit.ignore
    def apply_parameter_constraints(self):
        self.l1.apply_parameter_constraints()
        self.l2.apply_parameter_constraints()
        
    def forward(self, inputs):
        out = self.l1(inputs)
        self.l1_spike = out
        out = self.l2(out)
        self.l2_spike = out

        return out
    @torch.jit.ignore
    def forward_with_states(self, inputs):
        states = []
        s1 = self.l1.initial_state(inputs.shape[0], inputs.device)
        s1_list = []
        s2 = self.l2.initial_state(inputs.shape[0], inputs.device)
        s2_list = []
        out_sequence = []
        for t, x_t in enumerate(inputs.unbind(1)):
            out, s1 = self.l1.forward_cell(x_t, s1)
            s1_list.append(s1)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out, s2 = self.l2.forward_cell(out, s2)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            s2_list.append(s2)
            out_sequence.append(out)
        s1_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s1_list)], dim=0)
        states.append(s1_list)
        s2_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s2_list)], dim=0)
        states.append(s2_list)
        out = torch.stack(out_sequence, dim=1)
        return out, states
class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.light_decoder = cfg.light_decoder
        if self.light_decoder:
            self.forward = self.forward_light
            self.forward_with_states = self.forward_light_with_states
        else:
            self.l1 = layer_map[cfg.l1.cell](cfg.l1)
            self.forward = self.forward_full
            self.forward_with_states = self.forward_full_with_states
        self.l1_spike = torch.empty(size=())
        self.aux_out = torch.empty(size=())
        self.out_layer = LI(cfg.l_out)
        self.dropout = cfg.dropout
    @torch.jit.ignore
    def apply_parameter_constraints(self):
        if not self.light_decoder:
            self.l1.apply_parameter_constraints()
        self.out_layer.apply_parameter_constraints()
    
    def forward_light(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out_layer(inputs)
        
    def forward_full(self, inputs):
        out = self.l1(inputs)
        self.l1_spike = out
        out = self.out_layer(out)
        return out
    
    @torch.jit.ignore
    def forward_light_with_states(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor,]]:
        states = []
        s_out = self.out_layer.initial_state(inputs.shape[0], inputs.device)
        s_out_list = [s_out,]
        out_sequence = []
        for t, x_t in enumerate(inputs.unbind(1)):
            out, s_out = self.out_layer.forward_cell(x_t, s_out)
            s_out_list.append(s_out)
            out_sequence.append(out)
        s_out_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s_out_list)], dim=0)
        states.append(s_out_list)
        out = torch.stack(out_sequence, dim=1)
        return out, states
    @torch.jit.ignore
    def forward_full_with_states(self, inputs):
        states = []
        s1 = self.l1.initial_state(inputs.shape[0], inputs.device)
        s1_list = []
        s_out = self.out_layer.initial_state(inputs.shape[0], inputs.device)
        s_out_list = [s_out,]
        out_sequence = []
        for t, x_t in enumerate(inputs.unbind(1)):
            out, s1 = self.l1.forward_cell(x_t, s1)
            s1_list.append(s1)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out, s_out = self.out_layer.forward_cell(out, s_out)
            s_out_list.append(s_out)
            out_sequence.append(out)
        s1_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s1_list)], dim=0)
        states.append(s1_list)
        s_out_list = torch.stack([torch.stack(x, dim=-2) for x in zip(*s_out_list)], dim=0)
        states.append(s_out_list)
        out = torch.stack(out_sequence, dim=1)
        return out, states
    
class Net(torch.nn.Module): 
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)
    
    def forward(self, inputs: torch.Tensor):
        out = self.encoder(inputs)
        out = self.decoder(out)
        return out
    @torch.jit.ignore
    def apply_parameter_constraints(self):
        self.encoder.apply_parameter_constraints()
        self.decoder.apply_parameter_constraints()
        
    @torch.jit.ignore
    def forward_with_states(self, inputs: torch.Tensor):
        out, enc_states = self.encoder.forward_with_states(inputs)
        out, dec_states = self.decoder.forward_with_states(out)
        enc_states.extend(dec_states)
        return out, enc_states
        
         

class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        print(cfg)
        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.lr = cfg.lr
        self.prediction_delay = cfg.dataset.prediction_delay
        self.skip_first_n = cfg.skip_first_n

        # For learning rate scheduling (used for oscillation task)
        self.factor = cfg.factor
        self.patience = cfg.patience
        self.num_fast_epoch = cfg.get('num_fast_epoch', 0)
        self.fast_epoch_lr_factor = cfg.get('fast_epoch_lr_factor', 0)

        self.output_size = cfg.dataset.num_classes
        self.batch_size = cfg.dataset.batch_size
        self.model = Net(cfg) #, fullgraph=True, dynamic=False)#, example_inputs=[torch.zeros([256, 1024, 1], dtype=torch.float),])
        self.model = torch.compile(self.model, dynamic=True)
        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()
        self.loss = MultiScaleMelSpetroLoss(cfg.dataset.sampling_freq, cfg.n_mels, cfg.min_window, cfg.max_window)
        # regularization parameters
        self.min_spike_prob = cfg.min_spike_prob
        self.max_spike_prob = cfg.max_spike_prob
        self.min_layer_coeff = cfg.min_layer_coeff
        self.max_layer_coeff = cfg.max_layer_coeff
        self.light_decoder = cfg.decoder.light_decoder
        if self.light_decoder:
            self.min_layer_coeff = self.min_layer_coeff[:2]
            self.max_layer_coeff = self.max_layer_coeff[:2] 
        self.grad_norm = cfg.grad_norm
        self.automatic_optimization=False

    def forward(self, inputs: torch.Tensor):
        out = self.model(inputs)
        return out
    # 
    def forward_with_states(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out, states = self.model.forward_with_states(inputs)
        self.states = [s[:, :, self.prediction_delay:] for s in states]
        return out

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.model.apply_parameter_constraints()
        

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
        targets = targets[:, 1+self.skip_first_n:]
        loss = self.loss(outputs[:, self.skip_first_n+self.prediction_delay:], targets)
        block_idx = block_idx[:, self.prediction_delay:].unsqueeze(-1)
        outputs_reduce = outputs

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
        targets = targets[:, 1:]
        outputs = outputs[:, self.prediction_delay:]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
       
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
        opt_1, opt_2 = self.optimizers()
        if self.trainer.current_epoch < self.num_fast_epoch:
            opt = opt_1
        else:
            opt = opt_2
        
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
            
        sum_spikes = [self.model.encoder.l1_spike, self.model.encoder.l2_spike]
        if not self.light_decoder:
            sum_spikes.extend([self.model.decoder.l1_spike,])
        # remove ignored spike then take the neuron-wise spike proba 
        sum_spikes = [x[:, self.skip_first_n:].mean(0).mean(0) for x in sum_spikes]
        reg_upper, log_upper = snn_regularization(sum_spikes, self.max_spike_prob, torch.tensor(self.max_layer_coeff, device=inputs.device), 'upper', reduce_layer='sum')
        reg_lower, log_lower = snn_regularization(sum_spikes, self.min_spike_prob, torch.tensor(self.min_layer_coeff, device=inputs.device), 'lower', reduce_layer='sum')
        log_upper.update(log_lower)
        reg_loss = reg_upper + reg_lower
        
        self.log_dict(log_upper, prog_bar=True, on_epoch=True)
        opt.zero_grad()

        loss = loss + reg_loss
        self.manual_backward(loss)
        self.log_dict(grad_norm(self, norm_type=2), on_epoch=True)
        self.clip_gradients(opt, gradient_clip_val=self.grad_norm, gradient_clip_algorithm="norm")
        opt.step()
        if self.trainer.is_last_batch and self.trainer.current_epoch  > self.num_fast_epoch:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.tracking_metric])
            
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self.forward_with_states(inputs)
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
        
        if batch_idx == 0:
            
            tmp_block_idx = block_idx.clone()
            tmp_block_idx[:, :self.skip_first_n, :] = 0
            # determine a random example to visualized
            spike_probabilities = get_per_layer_spike_probs(
                self.states,
                tmp_block_idx,
            )
            rnd_batch_idx = torch.randint(0, inputs.shape[0], size=()).item()
            prev_layer_input = inputs[rnd_batch_idx]
            layers = [self.model.encoder.l1,self.model.encoder.l2,]
            if not self.light_decoder:
                # layers.extend([self.model.decoder.l1, self.model.decoder.l2, self.model.decoder.out_layer])
                layers.extend([self.model.decoder.l1, self.model.decoder.out_layer])
            else:
                layers.append(self.model.decoder.out_layer)
                
            for layer, module in enumerate(layers):
                if hasattr(module, "layer_stats"):
                    module.layer_stats(
                        logger=self.logger,
                        epoch_step=self.current_epoch,
                        inputs=prev_layer_input,
                        states=self.states[layer][:, rnd_batch_idx],
                        targets=targets[rnd_batch_idx],
                        layer_idx=layer,
                        block_idx=block_idx[rnd_batch_idx],
                        spike_probabilities=spike_probabilities[layer]
                        if len(spike_probabilities) > layer
                        else None,
                        output_size=self.output_size,
                        auto_regression=True
                    )
                    if layer < len(layers) - 1:
                        prev_layer_input = self.states[layer][1, rnd_batch_idx]

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self.forward_with_states(inputs)

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
        metrics = torchmetrics.MetricCollection(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                }
            )

        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")


    def configure_optimizers(self):
        opt_1 = torch.optim.Adam(params=self.parameters(), lr=self.lr*self.fast_epoch_lr_factor)
        opt_2 = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        lr_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt_2,
            mode=self.tracking_mode,
            factor=self.factor,
            patience=self.patience,
        )
        return (
            {
                "optimizer": opt_1
            },
            {
            "optimizer": opt_2,
            "lr_scheduler": {
                "scheduler": lr_2,
                "monitor": self.tracking_metric,
            },
        })
