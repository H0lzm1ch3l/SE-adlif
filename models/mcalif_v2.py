
from typing import Optional, Sequence

import torch._dynamo.guards
from functional.activations import SLAYER
from models.helpers import generic_scan, generic_scan_with_states, init_micheli_normal, spike_grad_injection_function
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter

from module.tau_trainers import TauTrainer, get_tau_trainer_class
from omegaconf import DictConfig
class MCAdLIF2(Module):
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
        self.dt = cfg.get('dt', 1.0)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = cfg.get('train_tau_u_method', 'interpolation')
        self.train_tau_w_method = cfg.get('train_tau_w_method', 'interpolation')
        self.train_tau_d_method = cfg.get('train_tau_d_method', 'interpolation')
        self.train_tau_t_method = cfg.get('train_tau_t_method', 'interpolation')
        self.recurrent_dendrite = cfg.get('recurrent_dendrite', False)
        self.unroll = cfg.get('unroll', 10)
        self.use_recurrent = cfg.get('use_recurrent', True)
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
        if isinstance(d_thr, Sequence):
            d_thr = torch.FloatTensor(self.out_features, device=device).uniform_(d_thr[0], d_thr[1])
        else:
            d_thr = torch.Tensor([d_thr,])
        if cfg.get('train_thr', False):
            self.s_thr = Parameter(s_thr)
            self.d_thr = Parameter(d_thr)
        else:
            self.register_buffer('s_thr', s_thr)
            self.register_buffer('d_thr', d_thr)
            
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.epsilon = cfg.get('epsilon', 0.05)
        
        self.tau_w_range = cfg.tau_w_range
        
        self.a_range = cfg.get('a_range', [0.0, 1.0])
        self.b_range = cfg.get('b_range',[0.0, 2.0]) 
        
        self.q = cfg.q

        # this is where I add the multi-compartment parameters
        # now what do i need
        # - the number of compartments
        # - the threshold for a dendritic compartment, for now just a scalar
        # - the dendritic current state variable
        # - the dendritic current decay
        # - value of the plateau potential
        # - tau for how long the plateau potential lasts

        self.num_compartments = cfg.get('num_compartments', 1)
        
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
            torch.empty((self.out_features + self.out_features * self.num_compartments, self.in_features), **factory_kwargs)
        )
        self.bias = Parameter(torch.empty(self.out_features + self.out_features * self.num_compartments, **factory_kwargs))
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
        
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_w_method)(
            self.out_features,
            self.dt, 
            self.tau_w_range[0], 
            self.tau_w_range[1],
            **factory_kwargs
        )
        self.d0 = Parameter(torch.empty((self.out_features, self.num_compartments), **factory_kwargs), requires_grad=False)

        self.tau_d_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_d_method)(
            self.out_features * self.num_compartments,
            self.dt,
            self.tau_d_range[0], 
            self.tau_d_range[1],
            **factory_kwargs,
        )
        self.tau_t_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_t_method)(
            self.out_features * self.num_compartments,
            self.dt,
            self.tau_t_range[0],
            self.tau_t_range[1],
            **factory_kwargs,
        )        
        self.a = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(self.out_features, **factory_kwargs))
        
        def step_fn(recurrent, alpha, beta, gamma, delta, s_thr, d_thr, a, b, u_rest, d_rest, carry, cur):
            u_tm1, z_tm1, w_tm1, d_tm1, t_tm1, p_tm1 = carry
            gamma = gamma.reshape(-1, self.num_compartments)
            delta = delta.reshape(-1, self.num_compartments)
            
            s_cur = cur[:, :self.num_out_neuron]
            d_cur = cur[:, self.num_out_neuron:].reshape(-1, self.num_out_neuron, self.num_compartments)
            
            if self.use_recurrent:
                cur_rec = F.linear(z_tm1, recurrent, None)
                s_cur = s_cur + cur_rec
                
            if self.recurrent_dendrite:
                cur_rec_d = torch.einsum('bni,nji->bnj', p_tm1, self.dendritic_recurrency_mask * self.dendritic_recurrent)
                d_cur = d_cur + cur_rec_d
            
            d = gamma * d_tm1 + (1.0 - gamma) * (d_cur)
            p = SLAYER.apply(d - d_thr, self.alpha, self.c)
            d = d * (1 - p.detach()) + (d_rest * p.detach())
            t = delta * t_tm1 + (1-delta) * p
            active_dendrite = SLAYER.apply(t - self.epsilon, self.alpha, self.c)
            d = (active_dendrite * self.u_p) + d

            u = alpha * u_tm1 + (1.0 - alpha) * (s_cur - w_tm1 + d.sum(-1))
            z = SLAYER.apply(u - s_thr, self.alpha, self.c)
            u = u * (1 - z.detach()) + (u_rest * z.detach())
            
            w = (beta * w_tm1 + (1.0 - beta) * (a * u + b * z) * self.q)
            return (u, z, w, d, t, p), z
        self.step = step_fn
        
        def wrapped_scan(u0: Parameter, z0: Tensor, w0: Parameter, d0: Parameter, t0: Parameter, x: Tensor,
                         recurrent: Parameter, alpha: Parameter, beta: Parameter, gamma: Parameter, delta: Parameter,
                         s_thr: Tensor, d_thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            d_rest = torch.zeros_like(u0)
                
            def wrapped_step(carry, s_cur, d_cur):
                return step_fn(recurrent, alpha, beta, gamma, delta, s_thr, d_thr, a, b, u_rest, d_rest, carry, s_cur, d_cur)

            return generic_scan(wrapped_step, (u0, z0, w0, d0, t0), x, self.unroll)
        
        def wrapped_scan_with_states(u0: Parameter, z0: Tensor, w0, d0: Parameter, t0: Parameter, x: Tensor,
                                      recurrent: Parameter, alpha: Parameter, beta: Parameter, gamma: Parameter, delta: Parameter,
                                      s_thr: Tensor, d_thr: Tensor, a: Parameter, b: Parameter):
            if self.use_u_rest:
                u_rest = u0
            else:
                u_rest = torch.zeros_like(u0)
            d_rest = torch.zeros_like(u0)

            def wrapped_step(carry, s_cur, d_cur):
                return step_fn(recurrent, alpha, beta, gamma, delta, s_thr, d_thr, a, b, u_rest, d_rest, carry, s_cur, d_cur)

            return generic_scan_with_states(wrapped_step, (u0, z0, w0, d0, t0), x, self.unroll)
        
        self.wrapped_scan = wrapped_scan
        self.wrapped_scan_with_states = wrapped_scan_with_states
        if not cfg.get('compile', False):
            self.wrapped_scan = torch.compiler.disable(self.wrapped_scan, recursive=False)
            self.wrapped_scan_with_states = torch.compiler.disable(self.wrapped_scan_with_states, recursive=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        self.tau_d_trainer.reset_parameters()
        self.tau_t_trainer.reset_parameters()
        
        decays = torch.cat((self.tau_u_trainer.get_decay(), self.tau_d_trainer.get_decay()), dim=0)
        M = torch.cat((torch.ones(self.out_features, device=decays.device), torch.ones(self.out_features * self.num_compartments, device=decays.device) * self.num_compartments), dim=0)
        init_micheli_normal(self.weight, threshold=self.d_thr, decay=decays, factor=M)
        
        torch.nn.init.zeros_(self.bias)
        
        # intra neuron recurrence init 
        if self.recurrent_dendrite:
            for i in range(self.num_out_neuron):
                torch.nn.init.orthogonal_(
                    self.dendritic_recurrent[i],
                    gain=1.0,
                )
                
        if self.train_u_p:
            # init_micheli_normal(self.u_p, threshold=torch.tensor(0.0), decay=self.tau_u_trainer.get_decay())
            torch.nn.init.zeros_(self.u_p)
        
        # h0 states 
        if self.train_u0:
            torch.nn.init.uniform_(self.u0, 0, self.thr[0].item())
        else:
            torch.nn.init.zeros_(self.u0)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(
                self.recurrent,
                gain=1.0,
            )
        
        torch.nn.init.uniform_(self.a, self.a_range[0], self.a_range[1])
        torch.nn.init.uniform_(self.b, self.b_range[0], self.b_range[1])
        
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
        w = torch.zeros(size=soma_size,
                        device=device, 
                        dtype=torch.float, 
                        pin_memory=None,
                        requires_grad=True
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
        return self.u0.unsqueeze(0), w, z, d, t, p

    # TODO: Adapt this to work with the new multi-compartmental LIF
    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.tau_d_trainer.apply_parameter_constraints()
        self.tau_t_trainer.apply_parameter_constraints()
        self.u0.data = self.u0 - torch.sign(self.u0)*torch.relu(torch.abs(self.u0) - self.s_thr)
        self.d0.data = self.d0 - torch.sign(self.d0)*torch.relu(torch.abs(self.d0) - self.d_thr)
        self.s_thr.data = torch.maximum(self.s_thr, torch.zeros_like(self.s_thr))
        self.d_thr.data = torch.maximum(self.d_thr, torch.zeros_like(self.d_thr))
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])
    
    def forward(self, input_tensor: Tensor, states: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        decay_d = self.tau_d_trainer.get_decay()
        decay_t = self.tau_t_trainer.get_decay()
        current = F.linear(input_tensor, self.weight, self.bias)
        new_states, z_t = self.step(self.recurrent, decay_u, decay_w, decay_d, decay_t, self.s_thr, self.d_thr, self.a, self.b, self.u0, self.d0, states, current)
        return z_t, new_states

    # TODO: Adapt this to work with the new multi-compartmental LIF
    def layer_forward(self, inputs: torch.Tensor) -> Tensor:
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        decay_d = self.tau_d_trainer.get_decay()
        decay_t = self.tau_t_trainer.get_decay()
        currents = F.linear(inputs, self.weight, self.bias)
        soma_current = currents[:, :self.num_out_neuron]
        dendritic_current = currents[:, self.num_out_neuron:].reshape(-1, self.num_out_neuron, self.num_compartments)
        u, z, w, d, t = self.initial_state(inputs.shape[0], inputs.device)
        out_buffer = self.wrapped_scan(u, z, w, d, t, soma_current, dendritic_current, self.recurrent, decay_u, decay_w, decay_d, decay_t, self.s_thr, self.d_thr, self.a, self.b)
        return out_buffer[:, :, :self.num_out_neuron]

    # TODO: Adapt this to work with the new multi-compartmental LIF
    @torch.no_grad()
    def layer_forward_with_states(self, inputs: torch.Tensor) -> Tensor:
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        decay_d = self.tau_d_trainer.get_decay()
        decay_t = self.tau_t_trainer.get_decay()
        currents = F.linear(inputs, self.weight, self.bias)
        soma_current = currents[:, :self.num_out_neuron]
        dendritic_current = currents[:, self.num_out_neuron:].reshape(-1, self.num_out_neuron, self.num_compartments)
        u, z, w, d, t = self.initial_state(inputs.shape[0], inputs.device)
        states, out_buffer = self.wrapped_scan_with_states(u, z, w, d, t, soma_current, dendritic_current, self.recurrent, decay_u, decay_w, decay_d, decay_t, self.s_thr, self.d_thr, self.a, self.b)
        return states[..., :self.num_out_neuron], out_buffer[..., :self.num_out_neuron]
    