from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from models.helpers import get_event_indices, save_distributions_to_aim, save_fig_to_aim, generic_scan
from module.tau_trainers import TauTrainer, get_tau_trainer_class
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from functools import partial
class LI(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        cfg: DictConfig,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(**kwargs)
        self.in_features = cfg.input_size
        self.out_features = cfg.dataset.num_classes
        self.dt = cfg.get('dt', 1.0)
        self.tau_u_range = cfg.tau_out_range
        self.train_tau_u_method = cfg.get('train_tau_out_method', 'fixed')
        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )

        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )
        self.unroll = cfg.get('unroll', 10)
        def step_fn(alpha, carry, x):
            u, = carry
            u = alpha * u + (1 - alpha)*x
            return (u,), u 
        def wrapped_scan(u0: Parameter, x: Tensor,
                         alpha: Parameter):
            _step_fn = partial(step_fn, alpha)
            return generic_scan(_step_fn, (u0, ), x, self.unroll)
        self.wrapped_scan = torch.compile(wrapped_scan)
        self.reset_parameters()
        
    @torch.compiler.disable
    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        torch.nn.init.uniform_(
            self.weight,
            -1 * torch.sqrt(1 / torch.tensor(self.in_features)),
            torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        torch.nn.init.zeros_(self.bias)

    @torch.compiler.disable
    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
        
    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> tuple[torch.Tensor,]:
        size = (batch_size, self.out_features)
        u = torch.zeros(size=size, 
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None
        )
        return (u,)
    def forward(self, inputs):
        current = F.linear(inputs, self.weight, self.bias)
        u, = self.initial_state(inputs.shape[0], device=inputs.device)
        decay_u = self.tau_u_trainer.get_decay()
        return self.wrapped_scan(u, current, decay_u)
    
    @torch.compiler.disable
    def forward_cell(self, input_tensor: Tensor, states: Tensor) -> Tuple[Tensor, Tensor]:
        u_tm1, = states
        decay_u = self.tau_u_trainer.get_decay()
        current = F.linear(input_tensor, self.weight, self.bias)
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * current
        return u_t, (u_t,)
    
    @torch.compiler.disable
    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        
    @torch.compiler.disable
    @staticmethod
    def plot_states(layer_idx, inputs, states, targets, block_idx, output_size, auto_regression):
        figure, axes = plt.subplots(nrows=3, ncols=1, sharex="all", figsize=(8, 11))
        is_events = torch.all(inputs == inputs.round())
        inputs = inputs.cpu().detach().numpy()
        # remove the first states as it's the initialization states
        states = states[:, 1:].cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        
        block_idx = block_idx.cpu().detach().numpy()
        targets_in_time = targets[1:]
        if auto_regression:
            targets_in_time = targets[1:]
        else:
            targets_in_time = targets[block_idx]
        
        if is_events:
            axes[0].eventplot(get_event_indices(inputs.T), color='black', orientation='horizontal')
        else:
            axes[0].plot(inputs)
        axes[0].set_ylabel("Input")
        axes[1].plot(states[0])
        axes[1].set_ylabel("v_t/output")
        if auto_regression:
            mse = ((states[0] - targets_in_time)**2).mean(-1)
            axes[2].plot(mse, color='blue', label='mse')
            x_min, x_max = axes[2].get_xlim()  
            x_half = (x_min + x_max) / 2  
            axes[2].axvline(x=x_half, color='red', linestyle='--', linewidth=2, label='Auto-regression start')
            axes[2].set_ylabel("MSE")
        else:
            pred = np.argmax(states[0], -1)
            axes[2].plot(pred, color="blue", label="Prediction")
            axes[2].plot(targets_in_time, color="red", label="Target")
            axes[2].set_ylabel("Class")
        axes[2].legend()
        figure.suptitle(f"Layer {layer_idx}\n")
        plt.tight_layout()
        plt.close(figure)
        return figure
    
    @torch.compiler.disable
    def layer_stats(
            self,
            layer_idx: int,
            logger,
            epoch_step: int,
            inputs: torch.Tensor,
            states: torch.Tensor,
            targets: torch.Tensor,
            block_idx: torch.Tensor,
            output_size: int,
            **kwargs,
        ):
            """Generate statistisc from the layer weights and a plot of the layer dynamics for a random task example
            Args:
                layer_idx (int): index for the layer in the hierarchy
                logger (_type_): aim logger reference
                epoch_step (int): epoch
                spike_probability (torch.Tensor): spike probability for each neurons
                inputs (torch.Tensor): random example
                states (torch.Tensor): states associated to the computation of the random example
                targets (torch.Tensor): target associated to the random example
                block_idx (torch.Tensor): block indices associated to the random example
            """
            save_fig_to_aim(
                logger=logger,
                name=f"{layer_idx}_Activity",
                figure=LI.plot_states(
                    layer_idx, inputs, states, targets, block_idx, output_size, auto_regression=kwargs['auto_regression']
                ),
                epoch_step=epoch_step,
            )

            distributions = [
                ("tau", self.tau_u_trainer.get_tau().cpu().detach().numpy()),
                ("weights", self.weight.cpu().detach().numpy()),
                ("bias", self.bias.cpu().detach().numpy()),
            ]

            save_distributions_to_aim(
                logger=logger,
                distributions=distributions,
                name=f"{layer_idx}",
                epoch_step=epoch_step,
            )
