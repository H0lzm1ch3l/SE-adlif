
import math
from typing import Optional, Sequence

import torch._dynamo.guards
from functional.activations import SLAYER, FastSigmoid, Sigmoid
from models.helpers import generic_scan, generic_scan_with_states, spike_grad_injection_function
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from scipy.stats import norm
from torch.nn.parameter import Parameter

from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig
class MCLIF2(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
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
        self.dt = cfg.get('dt', 1.0)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = cfg.get('train_tau_u_method', 'interpolation')
        self.train_tau_d_method = cfg.get('train_tau_d_method', 'interpolation')
        self.train_tau_t_method = cfg.get('train_tau_t_method', 'interpolation')
        self.unroll = cfg.get('unroll', 10)
        self.use_recurrent = cfg.get('use_recurrent', True)
        self.recurrent_dendrite = cfg.get('recurrent_dendrite', False)
        self.ff_gain = cfg.get('ff_gain', 1.0)
        s_thr = cfg.get('s_thr', 1.0)
        d_thr = cfg.get('d_thr', 1.0)
        self.num_out_neuron = cfg.get('num_out_neuron', self.out_features)
        self.use_u_rest = cfg.get('use_u_rest', False)
        self.train_u0 = cfg.get('train_u0', False)
        if isinstance(s_thr, Sequence):
            s_thr = torch.FloatTensor(self.out_features, device=device).uniform_(s_thr[0], s_thr[1])
        else:
            s_thr = torch.Tensor([s_thr,])
        if cfg.get('train_s_thr', False):
            self.s_thr = Parameter(s_thr)
        else:
            self.register_buffer('s_thr', s_thr)
        # self.s_thr = s_thr 
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.epsilon = cfg.get('epsilon', 0.5)
        self.num_compartments = cfg.get('num_compartments', 1)
        # soft reset parameters, since soft reset needs to be trained these are always parameters
        if cfg.get('use_soft_reset', True):
            self.u_reset = Parameter(torch.zeros((self.out_features), **factory_kwargs)) #  * cfg.get('u_reset', 1.0))   
            self.d_reset = Parameter(torch.zeros((self.out_features, self.num_compartments), **factory_kwargs)) #  * cfg.get('d_reset', 1.0))
        if isinstance(d_thr, Sequence):
            d_thr = torch.FloatTensor(self.out_features, device=device).uniform_(d_thr[0], d_thr[1])
        else:
            d_thr = torch.Tensor([d_thr,])
        if cfg.get('train_d_thr', False):
            self.d_thr = Parameter(d_thr)
        else:
            self.register_buffer('d_thr', d_thr)
        # self.d_thr = d_thr
            
        self.tau_d_range = cfg.tau_d_range
        self.tau_t_range = cfg.tau_t_range
        
        u_p = cfg.get('u_p', 0.5) / self.num_compartments # divide by num compartments to keep the total plateau potential constant
        # maybe calculate gain from cfg u_p value
        self.train_u_p = cfg.get('u_p_gain', True)
        if self.train_u_p:
            self.u_p = Parameter(torch.empty((self.out_features, self.num_compartments), **factory_kwargs))
            self.u_p_gain = u_p
        else:
            self.register_buffer("u_p", torch.empty(size=()))
            self.u_p = u_p

        self.weight = Parameter(
            torch.empty((self.out_features * self.num_compartments, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features * self.num_compartments, **factory_kwargs))
        if self.use_recurrent:
            self.recurrent = Parameter(
                    torch.empty((self.out_features, self.out_features), **factory_kwargs)
                )
        else:
            # registering an empty size tensor is required for the static analyser
            self.register_buffer("recurrent", torch.empty(size=()))
        # add weights for intra neuron dendritic recurrence
        if self.recurrent_dendrite:
            self.dendritic_recurrent = Parameter(
                    torch.empty((self.out_features, self.num_compartments, self.num_compartments), **factory_kwargs)
                )
            self.register_buffer('dendritic_recurrency_mask', 1 - torch.eye(self.num_compartments))
        else:
            # registering an empty size tensor is required for the static analyser
            self.register_buffer("dendritic_recurrent", torch.empty(size=()))
        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs, 
        )
        self.u0 = Parameter(torch.empty(self.out_features, **factory_kwargs), requires_grad=self.train_u0)
        self.tau_d_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_d_method)(
            self.out_features * self.num_compartments,
            self.dt,
            self.tau_d_range[0], 
            self.tau_d_range[1],
            **factory_kwargs,
        )
        self.d0 = Parameter(torch.empty((self.out_features, self.num_compartments), **factory_kwargs), requires_grad=False)

        self.tau_t_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_t_method)(
            self.out_features * self.num_compartments,
            self.dt,
            self.tau_t_range[0],
            self.tau_t_range[1],
            **factory_kwargs,
        )
        self.t0 = Parameter(torch.empty((self.out_features, self.num_compartments), **factory_kwargs), requires_grad=False)
        
        self.reset_parameters()
        def step_fn(alpha, beta, gamma, s_thr, d_thr, u_rest, d_rest, carry, cur):
            u_tm1, z_tm1, d_tm1, t_tm1, p_tm1 = carry
            beta = beta.reshape(-1, self.num_compartments)
            gamma = gamma.reshape(-1, self.num_compartments)
            cur = cur.reshape(-1, self.num_out_neuron, self.num_compartments)
            
            if self.use_recurrent:
                cur_rec_s = F.linear(z_tm1, self.recurrent, None)
                # cur = cur + cur_rec_s.reshape(-1, self.num_out_neuron, self.num_compartments)
            if self.recurrent_dendrite:
                cur_rec_d = torch.einsum('bni,nji->bnj', p_tm1, self.dendritic_recurrent) # self.dendritic_recurrency_mask * self.dendritic_recurrent)
                cur = cur + cur_rec_d
                
            d = beta * d_tm1 + (1.0 - beta) * cur
            p = SLAYER.apply(d - d_thr, self.alpha, self.c)
            
            t = gamma * t_tm1 + p
            active_dendrite = SLAYER.apply(t - self.epsilon, self.alpha, self.c)
            d -= self.d_reset * p.detach()
            d_influx = (d + active_dendrite) * self.u_p
                    
            u = alpha * u_tm1 + (1.0 - alpha) * (cur_rec_s + d_influx.sum(-1))
            z = SLAYER.apply(u - s_thr, self.alpha, self.c)
            u -= self.u_reset * z.detach()
            
            return (u, z, d, t, p), z
        self.step = step_fn
        
        def wrapped_scan(u0: Parameter, z0: Tensor, d0: Parameter, t0: Parameter, x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, gamma: Parameter,
                         s_thr: Tensor, d_thr: Tensor):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            d_rest = torch.zeros_like(u0)
                
            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, gamma, s_thr, d_thr, u_rest, d_rest, carry, cur)

            return generic_scan(wrapped_step, (u0, z0, d0, t0), x, self.unroll)
        
        def wrapped_scan_with_states(u0: Parameter, z0: Tensor, d0: Parameter, t0: Parameter, x: Tensor,
                                      recurrent: Parameter, alpha: Parameter, beta: Parameter, gamma: Parameter,
                                      s_thr: Tensor, d_thr: Tensor):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            d_rest = torch.zeros_like(u0)

            def wrapped_step(carry, cur):
                return step_fn(recurrent, alpha, beta, gamma, s_thr, d_thr, u_rest, d_rest, carry, cur)

            return generic_scan_with_states(wrapped_step, (u0, z0, d0, t0), x, self.unroll)
        
        self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states

    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        self.tau_d_trainer.reset_parameters()
        self.tau_t_trainer.reset_parameters()
        
        # self._leak_comp_adj_init()
        # Decay + Compartment adjusted Aurora Micheli Init @https://github.com/AuroraMicheli/Weight-Initialization-SNN:
        area_from_threshold_to_infinity = 1 - torch.distributions.normal.Normal(0, 1).cdf(self.d_thr)
        var_w_optimal = 1/(self.in_features*area_from_threshold_to_infinity) * torch.sqrt(1 - self.tau_d_trainer.get_decay())
        self.weight.data = torch.distributions.normal.Normal(0, torch.sqrt(var_w_optimal)).sample((self.in_features,)).T
       
        torch.nn.init.zeros_(self.bias)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )
        # intra neuron recurrence init 
        if self.recurrent_dendrite:
            for i in range(self.num_out_neuron):
                torch.nn.init.orthogonal_(
                    self.dendritic_recurrent[i],
                    gain=1.0,
                )
        if self.train_u_p:
            # we assume that s_thr is 0 because u_p is a factor for the plateau potential, in case we get one within the first steps of training this would otherwise 
            # blow up the network activity - these are dendritic weights, so normal dense weights
            var_w_optimal = 1/(self.num_compartments) * torch.sqrt(1 - self.tau_u_trainer.get_decay())
            self.u_p.data = torch.distributions.normal.Normal(0, torch.sqrt(var_w_optimal)).sample((self.num_compartments,)).T
            # torch.nn.init.zeros_(self.u_p)
        # h0 states 
        if self.train_u0:
            torch.nn.init.uniform_(self.u0, 0, self.s_thr[0].item())
        else:
            torch.nn.init.zeros_(self.u0)
    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> tuple[Tensor, Tensor]:
        soma_size = (batch_size, self.out_features)
        dend_size = (batch_size, self.out_features, self.num_compartments)
        z = torch.zeros(size=soma_size,
                        device=device,
                        dtype=torch.float,
                        layout=None, 
                        pin_memory=None
                        )
        d = torch.zeros(size=dend_size,
                        device=device,
                        dtype=torch.float,
                        layout=None,
                        pin_memory=None
                        )
        t = torch.zeros(size=dend_size,
                        device=device,
                        dtype=torch.float,
                        layout=None,
                        pin_memory=None
                        )
        p = torch.zeros(size=dend_size,
                        device=device,
                        dtype=torch.float,
                        layout=None,
                        pin_memory=None
                        )
        return self.u0.unsqueeze(0), z, d, t, p
    
    def forward(self, input_tensor: Tensor, states: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        decay_d = self.tau_d_trainer.get_decay()
        decay_t = self.tau_t_trainer.get_decay()
        current = F.linear(input_tensor, self.weight, self.bias)
        new_states, z_t = self.step(decay_u, decay_d, decay_t, self.s_thr, self.d_thr, self.u0, self.d0,  states, current)
        return z_t, new_states

    # TODO: Adapt this to work with the new multi-compartmental LIF
    def layer_forward(self, inputs: torch.Tensor) -> Tensor:
        decay_u = self.tau_u_trainer.get_decay()
        decay_d = self.tau_d_trainer.get_decay()
        decay_t = self.tau_t_trainer.get_decay()
        current = F.linear(inputs, self.weight, self.bias)
        u, z, d, t = self.initial_state(inputs.shape[0], inputs.device)
        out_buffer = self.wrapped_scan(u, z, d, t, current, self.recurrent, decay_u, decay_d, decay_t, self.s_thr, self.d_thr)
        return out_buffer[:, :, :self.num_out_neuron]

    # TODO: Adapt this to work with the new multi-compartmental LIF
    @torch.no_grad()
    def layer_forward_with_states(self, inputs: torch.Tensor) -> Tensor:
        decay_u = self.tau_u_trainer.get_decay()
        decay_d = self.tau_d_trainer.get_decay()
        decay_t = self.tau_t_trainer.get_decay()
        current = F.linear(inputs, self.weight, self.bias)
        u, z, d, t = self.initial_state(inputs.shape[0], inputs.device)
        states, out_buffer = self.wrapped_scan_with_states(u, z, d, t, current, self.recurrent, decay_u, decay_d, decay_t, self.s_thr, self.d_thr)
        return states[..., :self.num_out_neuron], out_buffer[..., :self.num_out_neuron]
    
    # TODO: Adapt this to work with the new multi-compartmental LIF
    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_d_trainer.apply_parameter_constraints()
        self.tau_t_trainer.apply_parameter_constraints()
        self.u0.data = self.u0 - torch.sign(self.u0)*torch.relu(torch.abs(self.u0) - self.s_thr)
        self.d0.data = self.d0 - torch.sign(self.d0)*torch.relu(torch.abs(self.d0) - self.d_thr)
        self.s_thr.data = torch.maximum(self.s_thr, torch.zeros_like(self.s_thr))
        self.d_thr.data = torch.maximum(self.d_thr, torch.zeros_like(self.d_thr))
