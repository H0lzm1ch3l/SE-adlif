from typing import Optional, Sequence, Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from models.helpers import SLAYER, get_event_indices, save_distributions_to_aim, save_fig_to_aim
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
        if isinstance(thr, Sequence):
            thr = torch.FloatTensor(self.out_features, device=device).uniform_(thr[0], thr[1])
        else:
            thr = torch.Tensor([thr,])
        self.register_buffer('thr', thr)
        self.alpha = cfg.get('alpha', 5.0)
        self.c = cfg.get('c', 0.4)
        self.tau_u_range = cfg.tau_u_range
        self.train_tau_u_method = 'interpolation'
        self.tau_w_range = cfg.tau_w_range
        self.train_tau_w_method = 'interpolation'        
        self.use_recurrent = cfg.get('use_recurrent', True)
        
        self.ff_gain = cfg.get('ff_gain', 1.0)
        self.a_range = [0.0, 1.0]
        self.b_range = [0.0, 2.0]
        
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
        torch.nn.init.zeros_(self.u0)
        torch.nn.init.zeros_(self.w0)
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
            pin_memory=None
        )
        w = torch.zeros(
            size=size,
            device=device, 
            dtype=torch.float, 
            layout=None, 
            pin_memory=None
        )
        return u, z, w
    @torch.jit.ignore
    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        
        out_buffer = torch.zeros(inputs.size(0), inputs.size(1), self.out_features, device=inputs.device)
        for i, cur in enumerate(current.unbind(1)):
            if self.use_recurrent:
                cur_rec = F.linear(z, self.recurrent, None)
                cur = cur + cur_rec
            
            u_t = decay_u * u + (1.0 - decay_u) * (
                cur - w
            )
            u_thr = u_t - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte)
            z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
            u_t = u_t * (1 - z_t.detach())
            w = (
                decay_w * w + (1.0 - decay_w) * (self.a * u + self.b * z) * self.q
                )
            z = z_t
            u = u_t
            out_buffer[:, i] = z
        return out_buffer
    
    @torch.jit.ignore
    def forward_cell(
        self, input_tensor: Tensor,  states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current
            
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
            soma_current - w_tm1
        )
        
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        u_t = u_t * (1 - z_t.detach())
        w_t = (
            decay_w * w_tm1
            + (1.0 - decay_w) * (self.a * u_tm1 + self.b * z_tm1) * self.q
        )
        return z_t, (u_t, z_t, w_t)
    
    @torch.jit.ignore
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
    @torch.jit.ignore
    def layer_stats(self, layer_idx: int, logger, epoch_step: int, spike_probabilities: torch.Tensor,
                    inputs: torch.Tensor, states: torch.Tensor, **kwargs):
        """Generate statistisc from the layer weights and a plot of the layer dynamics for a random task example
        Args:
            layer_idx (int): index for the layer in the hierarchy
            logger (_type_): aim logger reference
            epoch_step (int): epoch  
            spike_probability (torch.Tensor): spike probability for each neurons
            inputs (torch.Tensor): random example 
            states (torch.Tensor): states associated to the computation of the random example
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
                         ("bias", self.bias.cpu().detach().numpy())
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
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        
        out_buffer = torch.zeros(inputs.size(0), inputs.size(1), self.out_features, device=inputs.device)
        for i, cur in enumerate(current.unbind(1)):
            if self.use_recurrent:
                cur_rec = F.linear(z, self.recurrent, None)
                cur = cur + cur_rec
            
            u = decay_u * u + (1.0 - decay_u) * (
                cur - w
            )
            u_thr = u - self.thr
            # Forward Gradient Injection trick (credits to Sebastian Otte)
            z = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
            u = u * (1 - z.detach())
            w = (
                decay_w * w + (1.0 - decay_w) * (self.a * u + self.b * z) * self.q
                )
            out_buffer[:, i] = z
        return out_buffer
    @torch.jit.ignore
    def forward_cell(
        self, input_tensor: Tensor, states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        soma_current = F.linear(input_tensor, self.weight, self.bias)
        if self.use_recurrent:
            soma_rec_current = F.linear(z_tm1, self.recurrent, None)
            soma_current += soma_rec_current
            
        u_t = decay_u * u_tm1 + (1.0 - decay_u) * (
            soma_current - w_tm1
        )
        u_thr = u_t - self.thr
        # Forward Gradient Injection trick (credits to Sebastian Otte)
        z_t = torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach() + (u_thr - u_thr.detach()) * SLAYER(u_thr, self.alpha, self.c).detach()
        u_t = u_t * (1 - z_t.detach())
        
        w_t = (
            decay_w * w_tm1 + (1.0 - decay_w) * (self.a * u_t + self.b * z_t ) * self.q
        )
        return z_t, (u_t, z_t, w_t)
    