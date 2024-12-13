from functools import partial
from math import ceil
import math
from typing import Callable, Optional, Sequence, Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.helpers import get_event_indices, save_distributions_to_aim, save_fig_to_aim, spike_grad_injection_function, generic_scan, generic_scan_with_states
from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig
import matplotlib.pyplot as plt

class EFAdLIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    a: Tensor
    b: Tensor 
    weight: Tensor

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
        self.out_features = cfg.n_neurons
        self.dt =  cfg.get('dt', 1.0)
        thr = cfg.get('thr', 1.0)
        self.unroll = cfg.get('unroll', 10)
        if isinstance(thr, Sequence):
            thr = torch.FloatTensor(self.out_features, device=device).uniform_(thr[0], thr[1])
        else:
            thr = Tensor([thr,])
        if cfg.train_thr:
            self.thr = Parameter(thr)
        else:
            self.register_buffer('thr', thr)
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = 'interpolation'
        self.tau_w_range = cfg.tau_w_range
        self.train_tau_w_method = 'interpolation'        
        self.use_recurrent = cfg.get('use_recurrent', True)
        
        self.ff_gain = cfg.get('ff_gain', 1.0)
        self.a_range =  cfg.get('a_range', [0.0, 1.0])
        self.b_range = cfg.get('b_range',[0.0, 2.0])
        self.num_out_neuron = cfg.get('num_out_neuron', self.out_features)
        self.use_u_rest = cfg.get('use_u_rest', False)

        self.q = cfg.q
        
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
                self.out_features,
                self.dt, 
                self.tau_u_range[0], 
                self.tau_u_range[1],
                **factory_kwargs)
        
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_w_method)(
                self.out_features,
                self.dt, 
                self.tau_w_range[0], 
                self.tau_w_range[1],
                **factory_kwargs)
        
        
        self.weight = Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        
        if self.use_recurrent:
            self.recurrent = Parameter(
                torch.empty((self.out_features, self.out_features), **factory_kwargs)
            )
        else:
            # registering an empty size tensor is required for the static analyser when using jit.script
            self.register_buffer("recurrent", torch.empty(size=()))

        self.a = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.u0 = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.w0 = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.reset_parameters()
        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_tm1, z_tm1, w_tm1 = carry
            if self.use_recurrent:
                cur_rec = F.linear(z_tm1, recurrent, None)
                cur = cur + cur_rec
            
            u = alpha * u_tm1 + (1.0 - alpha) * (
                cur - w_tm1
            )
            u_thr = u - thr
            z = spike_grad_injection_function(u_thr, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest*z.detach()
            w = (
                beta * w_tm1 + (1.0 - beta) * (a * u_tm1 + b * z_tm1) * self.q
                )
            return (u, z, w), z
        def wrapped_scan(u0: Parameter, z0: Tensor, w0: Parameter, 
                         x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)

            return generic_scan(wrapped_step, (u0, z0, w0), x, self.unroll)
        def wrapped_scan_with_states(u0: Parameter, z0: Tensor, w0: Parameter, x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)

            return generic_scan_with_states(wrapped_step, (u0, z0, w0), x, self.unroll)
        if cfg.compile:
            self.wrapped_scan = torch.compile(wrapped_scan)
        else:
            self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states
    
    def reset_parameters(self) -> None:
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        
        
        torch.nn.init.uniform_(
            self.weight,
            -self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
            self.ff_gain * torch.sqrt(1 / torch.tensor(self.in_features)),
        )
        
        torch.nn.init.zeros_(self.bias)
        
        # h0 states 
        torch.nn.init.uniform_(self.u0, 0, self.thr.item())
        torch.nn.init.uniform_(self.w0, -1, 1)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )
        
        torch.nn.init.uniform_(self.a, self.a_range[0], self.a_range[1])
        torch.nn.init.uniform_(self.b, self.b_range[0], self.b_range[1])
        
    def initial_state(self, batch_size:int, device: Optional[torch.device] = None) -> tuple[Tensor, Tensor, Tensor]:
        size = (batch_size, self.out_features)
        u = torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None
        )
        z = torch.zeros(
            size=size, 
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None,
            requires_grad=True
        )
        w = torch.zeros(
            size=size,
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None,
            requires_grad=True
        )
        return self.u0.unsqueeze(0), z, w

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])
        self.u0.data = torch.clamp(self.u0, -self.thr, self.thr)
        
    def forward(self, inputs: Tensor) -> Tensor:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        out_buffer = self.wrapped_scan(u, z, w, current, self.recurrent, decay_u, decay_w, self.thr, self.a, self.b)
        return out_buffer[:, :, :self.num_out_neuron]

    @torch.no_grad()
    def forward_with_states(self, inputs) -> Tuple[Tensor, Tensor]:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        states, out = self.wrapped_scan_with_states(u, z, w, current, self.recurrent, decay_u, decay_w, self.thr, self.a, self.b)
        return states[..., :self.num_out_neuron], out[..., :self.num_out_neuron]
    
    @torch.compiler.disable
    @staticmethod
    def plot_states(layer_idx, inputs, states):
        figure, axes = plt.subplots(
        nrows=4, ncols=1, sharex='all', figsize=(8, 11))
        is_events = torch.all(inputs == inputs.round())

        inputs = inputs.cpu().detach().numpy()
        states = states.cpu().detach().numpy()
        if is_events:
            axes[0].eventplot(get_event_indices(inputs.T), color='black', orientation='horizontal')
        else:
            axes[0].plot(inputs)
        axes[0].set_ylabel('input')
        axes[1].plot(states[0])
        axes[1].set_ylabel("u_t")
        axes[2].plot(states[2])
        axes[2].set_ylabel("w_t")
        axes[3].eventplot(get_event_indices(states[1].T), color='black', orientation='horizontal')
        axes[3].set_ylabel("z_t/output")
        nb_spikes_str = str(states[1].sum())
        figure.suptitle(f"Layer {layer_idx}\n Nb spikes: {nb_spikes_str},")
        plt.close(figure)
        return figure
    
    @torch.compiler.disable
    def layer_stats(self, layer_idx: int, logger, epoch_step: int, spike_probabilities: Tensor,
                    inputs: Tensor, states: Tensor, **kwargs):
        """Generate statistisc from the layer weights and a plot of the layer dynamics for a random task example
        Args:
            layer_idx (int): index for the layer in the hierarchy
            logger (_type_): aim logger reference
            epoch_step (int): epoch  
            spike_probability (Tensor): spike probability for each neurons
            inputs (Tensor): random example 
            states (Tensor): states associated to the computation of the random example
        """

        save_fig_to_aim(
            logger=logger,
            name=f"{layer_idx}_Activity",
            figure=EFAdLIF.plot_states(layer_idx, inputs, states),
            epoch_step=epoch_step,
        )
        
        distributions = [("soma_tau", self.tau_u_trainer.get_tau().cpu().detach().numpy()),
                         ("soma_weights", self.weight.cpu().detach().numpy()),
                         ("adapt_tau", self.tau_w_trainer.get_tau().cpu().detach().numpy()),
                         ("spike_prob", spike_probabilities.cpu().detach().numpy()),
                         ("a", self.a.cpu().detach().numpy()),
                        ("b", self.b.cpu().detach().numpy()),
                         ("bias", self.bias.cpu().detach().numpy()),
                         ('u0', self.u0.cpu().numpy()),
                         ('w0', self.w0.cpu().numpy())
                        ]

        if self.use_recurrent:
            distributions.append(
                ("recurrent_weights", self.recurrent.cpu().detach().numpy())
            
            )
        save_distributions_to_aim(
            logger=logger,
            distributions=distributions,
            name=f"{layer_idx}",
            epoch_step=epoch_step,
        )
    
class SEAdLIF(EFAdLIF):
    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device, dtype, **kwargs)

        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_tm1, z_tm1, w_tm1 = carry
            if self.use_recurrent:
                cur_rec = F.linear(z_tm1, recurrent, None)
                cur = cur + cur_rec
            
            u = alpha * u_tm1 + (1.0 - alpha) * (
                cur - w_tm1
            )
            u_thr = u - thr
            z = spike_grad_injection_function(u_thr, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest*z.detach()
            w = (
                beta * w_tm1 + (1.0 - beta) * (a * u + b * z) * self.q
                )
            return (u, z, w), z

        def wrapped_scan(u0: Parameter, z0: Tensor, w0: Parameter, 
                          x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)
            return generic_scan(wrapped_step, (u0, z0, w0), x, self.unroll)
        def wrapped_scan_with_states(u0: Parameter, z0: Tensor, w0: Parameter, 
                          x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, 
                         thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur)
            return generic_scan_with_states(wrapped_step, (u0, z0, w0), x, self.unroll)
        
        if cfg.compile:
            self.wrapped_scan = torch.compile(wrapped_scan)
        else:
            self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states