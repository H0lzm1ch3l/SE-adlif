from torch.nn import Module
import torch
from torch.nn.parameter import Parameter

def get_tau_trainer_class(name: str):
    if name == "interpolation":
        return InterpolationTrainer
    elif name == 'interpolationExpSigmoid':
        return InterpolationExpSigmoidTrainer
    elif name == "sigmoid":
        return SigmoidTrainer
    elif name == "fixed":
        return FixedTau
    else:
        raise ValueError("Invalid tau trainer name: " + name)


class TauTrainer(Module):
    __constants__ = ["in_features"]
    weight: torch.Tensor
    def __init__(
        self,
        in_features: int,        
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs
        ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TauTrainer, self).__init__(**kwargs)
        self.dt = dt
        self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        self.register_buffer("tau_max", torch.tensor(tau_max, **factory_kwargs))
        self.register_buffer("tau_min", torch.tensor(tau_min, **factory_kwargs))
        
    def reset_parameters(self) -> None:
        raise NotImplementedError("This function should not be call from the base class.")

    def apply_parameter_constraints(self) -> None:
        raise NotImplementedError("This function should not be call from the base class.")
    
    def foward(self) -> torch.Tensor:
        raise  NotImplementedError("This function should not be call from the base class.")
    
    def get_tau(self) -> torch.Tensor:
        raise  NotImplementedError("This function should not be call from the base class.")
    
    def get_decay(self):
        return self.forward()


class FixedTau(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super(FixedTau, self).__init__(
            in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
    
    def apply_parameter_constraints(self):
        pass

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return self.weight

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, a=self.tau_min, b=self.tau_max)
        self.weight.requires_grad = False
        

class InterpolationTrainer(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(in_features, dt, tau_min, tau_max, device, dtype, **kwargs)

    def apply_parameter_constraints(self):
        with torch.no_grad():
            self.weight.clamp_(0.0, 1.0)

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return  self.weight * self.tau_max + (1.0 - self.weight) * self.tau_min

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        self.weight.requires_grad = True
        
class InterpolationExpSigmoidTrainer(TauTrainer):
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
        self.register_buffer('alpha_min',torch.exp(-dt/self.tau_min))
        self.register_buffer('alpha_max', torch.exp(-dt/self.tau_max))
        
    def apply_parameter_constraints(self):
        pass

    def forward(self):
        coef = torch.sigmoid(self.weight)
        return self.alpha_max * coef + (1.0 - coef)*self.alpha_min

    def get_tau(self):
        alpha = self.forward()
        
        return -1/torch.log(alpha)

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, -1, 1)
        self.weight.requires_grad = True


class SigmoidTrainer(TauTrainer):
    """
    Standard PLIF implementation. 
    Learns a decay factor beta in range (0, 1) using a raw sigmoid.
    Effectively allows tau to range from ~0 to Infinity.
    """
    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float = None, # Ignored, kept for compatibility
        tau_max: float = None, # Ignored, kept for compatibility
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        # We pass dummy values for min/max to the super class since we won't use them
        super().__init__(in_features, dt, tau_min, tau_max, device, dtype, **kwargs)
        
    def apply_parameter_constraints(self):
        # No clamping needed because Sigmoid handles the (0,1) constraint naturally
        pass

    def forward(self):
        # Returns the Decay Factor (beta) directly
        # Range: (0.0, 1.0)
        return torch.sigmoid(self.weight)

    def get_tau(self):
        # Inverse calculation: tau = -dt / ln(beta)
        # Added eps to avoid log(0) or div by zero errors
        beta = self.forward()
        return -self.dt / (torch.log(beta + 1e-8))

    def reset_parameters(self):
        # uniformly initialize taus in configured range
        random_taus = (
                torch.rand_like(self.weight) * (self.tau_max - self.tau_min) 
                + self.tau_min
            )
        
        # 1. Calculate target decay: beta = exp(-dt / tau)
        target_beta = torch.exp(-torch.tensor(self.dt) / random_taus)
        
        # 2. Inverse Sigmoid (Logit) to find the weight w
        # w = ln(beta / (1 - beta))
        init_weight = torch.log(target_beta / (1.0 - target_beta))
        
        # 3. Apply to weights (with a tiny bit of noise to break symmetry)
        with torch.no_grad():
            self.weight.data = init_weight.clone().detach()
            # Optional: Add small noise so neurons don't learn identically
            # self.weight.add_(torch.randn_like(self.weight) * 0.01)
        
        self.weight.requires_grad = True