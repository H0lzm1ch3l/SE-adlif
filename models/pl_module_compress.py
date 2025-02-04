import math
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig, open_dict
from pytorch_lightning.utilities import grad_norm
import matplotlib
import torchmetrics.audio
import os
from functional.metrics import MeanSquaredErrorFlat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functional.loss import (
    MultiResolutionSTFTLoss,
    get_per_layer_spike_probs,
    snn_regularization,
)
from models.alif import EFAdLIF, SEAdLIF
from models.helpers import A_law, inverse_A_law
from models.helpers import save_fig_to_aim
from models.li import LI
from models.sli import SLI
from models.lif import LIF
from models.rnn import LSTMCellWrapper


# from models.sli import SLI
torch.autograd.set_detect_anomaly(True)
torch._dynamo.config.cache_size_limit = 128
torch.set_float32_matmul_precision("high")
layer_map = {
    "lif": LIF,
    "se_adlif": SEAdLIF,
    "ef_adlif": EFAdLIF,
    "lstm": LSTMCellWrapper,
    "li": LI,
    "sli": SLI,
}


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = layer_map[cfg.l1.cell](cfg.l1)
        self.l2 = layer_map[cfg.l2.cell](cfg.l2)
        self.l1_spike = torch.empty(size=())
        self.l2_spike = torch.empty(size=())
        self.dropout = cfg.dropout

    def apply_parameter_constraints(self):
        self.l1.apply_parameter_constraints()
        self.l2.apply_parameter_constraints()

    def forward(self, inputs):
        out = self.l1(inputs)
        self.l1_spike = out
        out = self.l2(out)
        self.l2_spike = out

        return out

    @torch.no_grad()
    def forward_with_states(self, inputs):
        l1_states, out = self.l1.forward_with_states(inputs)
        l2_states, out = self.l2.forward_with_states(out)
        return [l1_states, l2_states], out


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.light_decoder = cfg.light_decoder
        self.l1 = layer_map[cfg.l1.cell](cfg.l1)
        self.l2 = layer_map[cfg.l2.cell](cfg.l2)

        self.l1_spike = torch.empty(size=())
        self.l2_spike = torch.empty(size=())

        self.aux_out = torch.empty(size=())
        self.out_layer = layer_map[cfg.l_out.cell](cfg.l_out)
        self.dropout = cfg.dropout

    def apply_parameter_constraints(self):
        self.l1.apply_parameter_constraints()
        if not self.light_decoder:
            self.l2.apply_parameter_constraints()
        self.out_layer.apply_parameter_constraints()

    def forward(self, inputs):
        out = inputs
        out = self.l1(out)
        self.l1_spike = out

        if not self.light_decoder:
            out = self.l2(out)
            self.l2_spike = out
        out = self.out_layer(out)
        return out

    def forward_with_states(self, inputs):
        out = inputs
        states = []
        l1_states, out = self.l1.forward_with_states(out)
        self.l1_spike = out
        states.append(l1_states)
        if not self.light_decoder:
            l2_states, out = self.l2.forward_with_states(out)
            self.l2_spike = out
            states.append(l2_states)
        out_states, out = self.out_layer.forward_with_states(out)
        states.append(out_states)
        return states, out


class Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)

    def forward(self, inputs: torch.Tensor):
        out = self.encoder(inputs)
        out = self.decoder(out)
        return out

    def apply_parameter_constraints(self):
        self.encoder.apply_parameter_constraints()
        self.decoder.apply_parameter_constraints()

    def forward_with_states(self, inputs: torch.Tensor):
        enc_states, out = self.encoder.forward_with_states(inputs)
        dec_states, out = self.decoder.forward_with_states(out)
        enc_states.extend(dec_states)
        return enc_states, out


class GenerativeSpectralLoss(torch.nn.Module):
    def __init__(
        self,
        num_bins,
        temp,
        gen_loss_gain,
        spectral_loss,
        spectral_loss_gain,
        discretization: str = "log",
        temp_decay: float = 1.0,
        min_temp: float = 1.0,
        transition_begin: int = 0,
        transition_steps: int = 1,
        
        
        *args,
        **kwargs,
    ):
        # discretization consider if num_bins levels are uniformly distributed
        # on a linear space or first mapped into a log space determined
        # by a standard A-law
        # In theory the log space is more suitable for audio perception
        super().__init__(*args, **kwargs)
        self.num_bins = num_bins
        # the discretization assume [-1, 1] normalization
        self.register_buffer("temp", torch.tensor(temp))
        self.a = torch.tensor(87.6)
        self.gen_loss_gain = gen_loss_gain
        self.spectral_loss = spectral_loss
        self.spectral_loss_gain = spectral_loss_gain
        self.discretization = discretization
        delta = torch.tensor(2.0 / (num_bins - 1))
        bin_edges = torch.tensor(
            [-1.0 + i * 2.0 / (num_bins - 1) for i in range(num_bins)]
        )
        self.register_buffer("delta", delta)  # uniform bining
        self.temp_decay = temp_decay
        self.register_buffer('min_temp', torch.tensor(min_temp))
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("batch_count", torch.tensor(0))
        self.transition_begin = transition_begin
        self.transition_steps = transition_steps

        def linear_quantize(x):
            return torch.round((x + 1.0) / self.delta).long()

        def log_quantize(x):
            return linear_quantize(A_law(x, self.a))

        def linear_dequantize(x):
            return torch.sum(x * self.bin_edges.view(1, 1, -1), dim=-1, keepdim=True)

        def log_dequantize(x):
            y = linear_dequantize(x)
            return inverse_A_law(y, self.a)

        self.quantize = (
            linear_quantize if self.discretization == "linear" else log_quantize
        )
        self.de_quantize = (
            linear_dequantize if self.discretization == "linear" else log_dequantize
        )

        # take the centers of the bin edges
        # used for reconstruction of the quantized signal
        self.gen_loss = torch.nn.CrossEntropyLoss()
    #TODO: use cond as get_temp have a control flow it is not compilable
    
    @torch.compiler.disable
    def get_temp(self, forced_temp=None):
        if forced_temp is None:   
            if self.transition_begin <= self.batch_count:
                rate_factor = ((self.batch_count - self.transition_begin) / self.transition_steps)
                decayed_temp = self.temp * (self.temp_decay ** rate_factor)
                if decayed_temp >= self.min_temp:
                    return decayed_temp
                else:
                    return self.min_temp
            else:
                return self.temp
        else:
            return forced_temp
        
    def generate_wave(self, outputs, temp):
        probs = torch.softmax(outputs/temp, -1)
        # 3. reconstructs from bins centers
        output_wave = self.de_quantize(probs)
        return output_wave
    
    def forward(self, outputs, targets):
        temp = self.get_temp()
        return self.forward_with_temp(outputs, targets, temp)

    def forward_with_temp(self, outputs, targets, temp):
        # outputs is assumed to be logits of shape (B, T, N) with N = num_bins
        # 1. compute sparse cross entropy w.r.t next token prediction
        bin_indices = self.quantize(targets)
        # soft_target = torch.softmax(-torch.square(bin_indices[:, 1:].unsqueeze(-1) - torch.arange(self.num_bins, device=outputs.device, dtype=torch.long).view(1, 1, -1))/0.1, dim=-1)
        # next token prediction        
        loss_gen = self.gen_loss(
            outputs[:, :-1].reshape(-1, self.num_bins), bin_indices[:, 1:].reshape(-1)
        )
        # 2. Compute differentiable sample using Gumbel softmax reparametrization trick on categorical distribution
        # hard = True will return one-hot vector, we want something softer
        # temp is softmax temperature
        # temp -> +inf => probs is uniform, tmp-> 0, probs is pure categorical distribution (one-hot)
        # probs always sum to 1
        probs = torch.softmax(outputs/temp, -1)
        # 3. reconstructs quantized vector from convex sum of bins centers
        output_wave = self.de_quantize(probs)
        # 4. compute spectral loss, recall that output_wave is the next step prediction
        # spectral loss expect (B, C, T) with C channel dim (in our case C=1)
        loss_spectral = self.spectral_loss(output_wave[:, :-1].permute((0,2,1)), targets[:, 1:].permute((0,2,1)))
        return self.gen_loss_gain * loss_gen + self.spectral_loss_gain * loss_spectral


class CompositeLoss(torch.nn.Module):
    def __init__(self, spectral_loss, spectral_loss_gain, mse_loss, mse_loss_gain):
        super().__init__()
        self.spectral_loss = spectral_loss
        self.mse_loss = mse_loss
        self.spectral_loss_gain = spectral_loss_gain
        self.mse_loss_gain = mse_loss_gain

    def forward(self, outputs, targets):
        spectral_loss = self.spectral_loss(outputs, targets)
        mse_loss = self.mse_loss(outputs, targets)
        return self.spectral_loss_gain * spectral_loss + self.mse_loss_gain * mse_loss


class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        print(cfg)
        if cfg.loss.type == "nll_spectral":
            # make sure that we don't average over the li / sli dim
            cfg.decoder.l_out.reduce = "none"

        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.lr = cfg.lr
        self.prediction_delay = cfg.dataset.prediction_delay
        self.skip_first_n = cfg.skip_first_n

        # For learning rate scheduling (used for oscillation task)
        self.factor = cfg.factor
        self.patience = cfg.patience
        self.num_fast_batch = cfg.get("num_fast_batch", 0)
        self.fast_batch_lr_factor = cfg.get("fast_batch_lr_factor", 0)
        self.min_lr = cfg.get("min_lr", 0)
        # This control the percentage of increase (or decrease) that should happend for 
        # a epoch to be consider good (default is 1%), see reduceLROnPlateau  threshold parameter.
        self.plateau_threshold = cfg.get('plateau_threshold', 1e-2)

        self.batch_size = cfg.dataset.batch_size
        self.model = Net(cfg)
        if cfg.get("compile", False):
            self.model = torch.compile(self.model, dynamic=True,)

        self.output_func = cfg.get("loss_agg", "softmax")
        self.init_metrics_and_loss()
        self.save_hyperparameters()
        
        windows_length = [2**i for i in range(cfg.loss.min_window, cfg.loss.max_window+1)]
        windows_hops = [w//4 for w in windows_length]

        # this is pretty standard that n_fft = windows_length, 
        # if n_fft < windows_length we loose information
        # if n_fft > windows_length the frequencies are oversampled/interpolated, 
        # As the delta freq = sampling_freq/win_length,
        # Audio framework recommands:
        # n_fft = windows_length*2**i
        # https://support.ircam.fr/docs/AudioSculpt/3.0/co/FFT%20Size.html
        # Issues for us, we need:
        # n_fft >> n_mels as low value imply empty filter 
        # n_fft/2 <= sample_length -  skip_first_n
        n_fft = [2**cfg.loss.max_window for w in windows_length]
        scale = cfg.loss.spectrum
        if scale == 'stft':
            scale = None
        w_log_mag = cfg.loss.get('w_log_mag','window')
        if w_log_mag == "window":
            w_log_mag = [math.sqrt(w/2) for w in windows_length]
        
        # norm convert "none" to None
        norm=cfg.loss.get('norm', 'slaney')
        if norm == 'none':
            norm = None
        spectral_loss = MultiResolutionSTFTLoss(
            fft_sizes=n_fft,
            win_lengths=windows_length,
            hop_sizes=windows_hops,
            w_sc=cfg.loss.get('w_sc', 0.0),
            w_log_mag=w_log_mag,
            w_lin_mag=cfg.loss.get('w_lin_mag', 1.0),
            w_phs=cfg.loss.get('phase_loss_coef', 0.0),
            sample_rate=cfg.dataset.sampling_freq,
            scale=scale,
            # this is ignore if scale is None
            n_bins=cfg.loss.n_mels,
            perceptual_weighting=cfg.loss.get('perceptual_weighting', False),
            scale_invariance=cfg.loss.get('scale_invariance', False),
            mag_distance=cfg.loss.get('mag_distance', "L1"),
            mag_distance_log=cfg.loss.get('mag_distance_log', "L2"),
            mel_scale=cfg.loss.get('mel_scale', 'slaney'),
            norm=norm,
            )

        if cfg.loss.type == "mse_spectral":
            self.loss = CompositeLoss(
                spectral_loss,
                cfg.loss.spectral_loss_gain,
                MSELoss(),
                cfg.loss.mse_loss_gain,
            )
        elif cfg.loss.type == "nll_spectral":
            self.loss = GenerativeSpectralLoss(
                num_bins=cfg.decoder.l_out.n_neurons,
                temp=cfg.loss.temp,
                gen_loss_gain=cfg.loss.mse_loss_gain,
                spectral_loss=spectral_loss,
                spectral_loss_gain=cfg.loss.spectral_loss_gain,
                discretization=cfg.loss.discretization,
                temp_decay=cfg.loss.get("temp_decay", 1.0),
                min_temp=cfg.loss.get("min_temp", 1.0),
                transition_begin=cfg.loss.get('transition_begin', 0),
                transition_steps=cfg.loss.get('transition_steps', 1)
            )
        else:
            self.loss = spectral_loss
        # if cfg.get("compile", False):
        #    self.loss = torch.compile(self.loss, dynamic=True,)
        # regularization parameters
        self.min_spike_prob = cfg.min_spike_prob
        self.max_spike_prob = cfg.max_spike_prob
        self.min_layer_coeff = cfg.min_layer_coeff
        self.max_layer_coeff = cfg.max_layer_coeff
        self.light_decoder = cfg.decoder.light_decoder
        if self.light_decoder:
            self.min_layer_coeff = self.min_layer_coeff[:-1]
            self.max_layer_coeff = self.max_layer_coeff[:-1]
        self.grad_norm = cfg.grad_norm
        self.automatic_optimization = False
        

    def forward(self, inputs: torch.Tensor):
        out = self.model(inputs)
        return out

    @torch.no_grad()
    def forward_with_states(
        self, inputs: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        states, out = self.model.forward_with_states(inputs)
        states = [s[:, :, self.prediction_delay :] for s in states]
        return states, out

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.model.apply_parameter_constraints()
        
        self.loss.batch_count += 1
    # on_train_epoch_end will have the val_{metric} https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks 
    def on_train_epoch_end(self):
        if (self.loss.batch_count > self.num_fast_batch):
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.tracking_metric])
        

    def process_predictions_and_compute_losses(self, outputs, targets):
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
        targets = targets[:, 1 + self.skip_first_n :]
        loss = self.loss(
            outputs[:, self.skip_first_n + self.prediction_delay :], targets
        )
        # block_idx = block_idx[:, self.prediction_delay :].unsqueeze(-1)
        # outputs_reduce = outputs[:, self.skip_first_n + self.prediction_delay :]

        return loss

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
        targets = targets[:, 1 + self.skip_first_n :]

        outputs = outputs[:, self.skip_first_n + self.prediction_delay :]
        if isinstance(self.loss, GenerativeSpectralLoss):
            outputs = self.loss.generate_wave(outputs, self.loss.get_temp())
        # outputs_flat = outputs.reshape(-1, outputs.shape[-1])
        # targets_flat = targets.reshape(-1, targets.shape[-1])

        metrics(outputs[:,:-1].permute((0,2,1))[:,0], targets[:, 1:].permute((0,2,1))[:,0])
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
        self.log(
            f"{prefix}spectral_loss",
            # spectral loss expect (B, C, T) with C channel dim (in our case C=1)
            self.loss.spectral_loss(outputs[:,:-1].permute((0,2,1)), targets[:, 1:].permute((0,2,1))),
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )

    def training_step(self, batch, batch_idx):
        opt_1, opt_2 = self.optimizers()
        if self.loss.batch_count < self.num_fast_batch:
            opt = opt_1
        else:
            opt = opt_2

        inputs, targets, block_idx = batch
        outputs = self(
            inputs,
        )
        loss = self.process_predictions_and_compute_losses(outputs, targets)
        self.update_and_log_metrics(
            outputs,
            targets,
            loss,
            self.train_metric,
            prefix="train_",
        )

        sum_spikes = [
            self.model.encoder.l1_spike,
            self.model.encoder.l2_spike,
            self.model.decoder.l1_spike,
        ]
        if not self.light_decoder:
            sum_spikes.extend(
                [
                    self.model.decoder.l2_spike,
                ]
            )
        # remove ignored spike then take the neuron-wise spike proba
        sum_spikes = [x[:, self.skip_first_n :].mean(0).mean(0) for x in sum_spikes]
        reg_upper, log_upper = snn_regularization(
            sum_spikes,
            self.max_spike_prob,
            torch.tensor(self.max_layer_coeff, device=inputs.device),
            "upper",
            reduce_layer="sum",
            reduce_neuron="sum",
            
        )
        reg_lower, log_lower = snn_regularization(
            sum_spikes,
            self.min_spike_prob,
            torch.tensor(self.min_layer_coeff, device=inputs.device),
            "lower",
            reduce_layer="sum",
        )
        log_upper.update(log_lower)
        reg_loss = reg_upper + reg_lower

        self.log_dict(log_upper, prog_bar=True, on_epoch=True)
        opt.zero_grad()

        loss = loss + reg_loss
        self.manual_backward(loss)
        self.log_dict(grad_norm(self, norm_type=2), on_epoch=True)
        self.clip_gradients(
            opt, gradient_clip_val=self.grad_norm, gradient_clip_algorithm="norm"
        )
        self.check_gradient(opt)
        self.log(
            "spectral_loss_temp",
            self.loss.get_temp(),
            on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        states, outputs = self.forward_with_states(inputs)
        loss = self.process_predictions_and_compute_losses(outputs, targets)

        self.update_and_log_metrics(
            outputs,
            targets,
            loss,
            self.val_metric,
            prefix="val_",
        )
        self.log(
            "val_spectral_loss_temp",
            self.loss.get_temp(),
            on_step=True, on_epoch=True)

        if batch_idx == 0:
            block_idx = block_idx[:, self.prediction_delay :].unsqueeze(-1)
            tmp_block_idx = block_idx.clone()
            tmp_block_idx[:, : self.skip_first_n, :] = 0
            # determine a random example to visualized
            # remove the last layer states as it is assumed to be non-spiking
            spike_probabilities = get_per_layer_spike_probs(
                states[:-1],
                tmp_block_idx,
            )
            rnd_batch_idx = torch.randint(0, inputs.shape[0], size=()).item()
            prev_layer_input = inputs[rnd_batch_idx]
            layers = [
                self.model.encoder.l1,
                self.model.encoder.l2,
                self.model.decoder.l1,
            ]
            if not self.light_decoder:
                # layers.extend([self.model.decoder.l1, self.model.decoder.l2, self.model.decoder.out_layer])
                layers.extend([self.model.decoder.l2, self.model.decoder.out_layer])
            else:
                layers.append(self.model.decoder.out_layer)

            batch_output = outputs[rnd_batch_idx]
            if isinstance(self.loss, GenerativeSpectralLoss):
                batch_output = self.loss.generate_wave(
                    batch_output.unsqueeze(0), self.loss.get_temp()
                ).squeeze()

            for layer, module in enumerate(layers):
                if hasattr(module, "layer_stats"):
                    module.layer_stats(
                        logger=self.logger,
                        epoch_step=self.current_epoch,
                        inputs=prev_layer_input,
                        states=states[layer][:, rnd_batch_idx],
                        targets=targets[rnd_batch_idx],
                        layer_idx=layer,
                        block_idx=block_idx[rnd_batch_idx],
                        spike_probabilities=spike_probabilities[layer]
                        if len(spike_probabilities) > layer
                        else None,
                        output_size=self.output_size,
                        auto_regression=True,
                        output=batch_output[self.prediction_delay :].cpu().numpy(),
                    )
                    if layer < len(layers) - 1:
                        prev_layer_input = states[layer][1, rnd_batch_idx]

            self.plot_reconstruction(batch_output, targets[rnd_batch_idx])
        return loss

    @torch.compiler.disable
    def plot_reconstruction(self, outputs, targets):
        figure, axes = plt.subplots(nrows=2, ncols=1, sharex="all", figsize=(8, 8))
        targets = torch.flatten(targets[1 + self.skip_first_n :])
        outputs = torch.flatten(outputs[self.skip_first_n + self.prediction_delay :])
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        axes[0].plot(outputs, label="Output")
        axes[0].set_ylabel("Reconstruction vs Target")

        axes[0].plot(targets, label="Target")
        axes[0].legend()

        axes[1].plot((outputs - targets) ** 2, label="MSE")
        axes[1].set_ylabel("MSE")
        plt.tight_layout()
        plt.close(figure)
        save_fig_to_aim(
            logger=self.logger,
            name=f"Reconstruction",
            figure=figure,
            epoch_step=self.current_epoch,
        )

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self.forward(inputs)
        loss = self.process_predictions_and_compute_losses(outputs, targets)

        self.update_and_log_metrics(
            outputs,
            targets,
            loss,
            self.test_metric,
            prefix="test_",
        )

        return loss

    def init_metrics_and_loss(self):
        metrics = torchmetrics.MetricCollection(
            {
                "mse": MeanSquaredErrorFlat(),
                "si_snr": torchmetrics.audio.ScaleInvariantSignalNoiseRatio(),
            }
        )
        

        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")


    def configure_optimizers(self):
        opt_1 = torch.optim.Adam(
            params=self.parameters(), lr=self.lr * self.fast_batch_lr_factor
        )
        opt_2 = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        lr_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt_2,
            mode=self.tracking_mode,
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            threshold=self.plateau_threshold,
        )
        return (
            {"optimizer": opt_1},
            {
                "optimizer": opt_2,
                "lr_scheduler": {
                    "scheduler": lr_2,
                    "monitor": self.tracking_metric,
                },
            },
        )
    @torch.compiler.disable
    def check_gradient(self, optimizer):
        valid_gradients = True
        un_valid_name = ""
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                # valid_gradients = not (torch.isnan(param.grad).any())
                if not valid_gradients:
                    un_valid_name = name
                    break
        if valid_gradients:
            optimizer.step()
        else:
            print(f"\ndetected inf or nan values in gradients at {un_valid_name}. not updating model parameters")
            optimizer.zero_grad()
