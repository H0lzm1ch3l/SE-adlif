from functools import partial
import logging
from typing import Callable
from omegaconf import DictConfig
import torch
import math

# SLAYER surrogate gradient function
def SLAYER(x: torch.Tensor, alpha: float, c: float) -> torch.Tensor:
    return c * alpha / (2 * torch.exp(x.abs() * alpha))

def spike_grad_injection_function(x: torch.Tensor, alpha: float, c: float) -> torch.Tensor:
    # Forward Gradient Injection trick (credits to Sebastian Otte)
    return torch.heaviside(x, torch.as_tensor(0.0).type(x.dtype)).detach() + (x - x.detach()) * SLAYER(x, alpha, c).detach()

# Decay adjusted Aurora Micheli Init @https://github.com/AuroraMicheli/Weight-Initialization-SNN:
def init_micheli_normal(tensor: torch.Tensor, threshold: torch.Tensor = torch.tensor(1.0), decay: float = None, factor: torch.Tensor = torch.tensor(1.0)):
    with torch.no_grad():
        area_from_threshold_to_infinity = 1 - torch.distributions.normal.Normal(0, 1).cdf(threshold)
        if decay is None:
            var_w_optimal = 1/(tensor.shape[1]*area_from_threshold_to_infinity*float(factor))
            torch.nn.init.normal_(tensor, 0, math.sqrt(var_w_optimal))
        else:
            var_w_optimal = 1/(tensor.shape[1]*area_from_threshold_to_infinity*factor)
            var_w_optimal = var_w_optimal * (1 - decay)
            tensor.data = torch.distributions.normal.Normal(0, torch.sqrt(var_w_optimal)).sample((tensor.shape[1],)).T
            
def init_simple_uniform(tensor: torch.Tensor, ff_gain: float, axis: int = -1):
    n = tensor.shape[axis]
    torch.nn.init.uniform_(
        tensor,
        -ff_gain  * torch.sqrt(1 / torch.tensor(n)),
        ff_gain * torch.sqrt(1 / torch.tensor(n)),
    )

def generic_scan(
    f: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[tuple[torch.Tensor, ...], torch.Tensor]], # f(s_t, x) -> (s_t+1, y)
    init: tuple[torch.Tensor, ...],
    xs: torch.Tensor,
    unroll: int = 1,
) -> torch.Tensor:
    """
        Create a scan like procedure that can be optimized by torch.compile.
        Code was lifted from https://github.com/pytorch/pytorch/issues/50688#issuecomment-2315002649 (SamPruden)
        
        What the code do:
        This is a pseudo scan function https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
        
        A scan in a higher-order function that loops over a statefull function from an initial state and an input list:
        
        def scan(f, init, xs):
            carry = init
            ys = []
            for x in xs:
                carry, y = f(carry, x)
                ys.append(y)
            return carry, np.stack(ys)
        
        A scan has the possibility to unroll in such a way that 
        that k iterations of f are made inside the loop instead of 1.
        
        This effectively creates two loops.
        def scan(f, init, xs, unroll=1):
            if unroll == 1: # do normal scan
            
            num_chunk = math.ceil(xs/unroll) 
            xs_chunk = np.split_array(xs, num_chunk)
            
            carry = init
            ys = []
            for chunk in xs_chunk: # outer loop
                y_chunk =[]
                for x in chunk: # inner loop
                    carry, y = f(carry, x)
                y_chunk.append(y)
                ys.extends(y_chunk)
            return carry, np.stack(ys)
        
        In this code, only the inner loop is compiled, the outer loop is kept in non-optimised code.
        The code is written so that ys is buffered.
        
         
        Reasoning:
        By default torch.compile will fully unroll each loop in the computation flow.
        This is not ideal for RNN where the number of iterations in the loop can be large.
        As such, the intermediate representation may require a large amount of virtual registers that cannot be matched to the hardware registers. (I assume, I'm not a low level guy.) 
        Also, unrolling means more instructions (larger binary) need to be stored to the device. 
        This (again, I assume) can lead to less room for data and memory spills/cache misses, slowing down the computation instead of improving it.
        Unrolling is generally a good thing (better cache locality, pipelining, out-of-order execution, reduced loop overhead), 
        but ideally the depth of unrolling should match the CPU/GPU capacity.

        Args:
            f (Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[tuple[torch.Tensor, ...], torch.Tensor]]) step function
            init (tuple[torch.Tensor, ...]): intial carry/states
            xs (torch.Tensor): inputs tensor
            unroll (int, optional): unrolling factor. Defaults to 1.

        Returns:
            torch.Tensor: output tensor
        """
    init_carry = init
    num_chunk = math.ceil(xs.shape[1] / unroll)
    out_ys = torch.empty_like(xs)
    
    def unrolled_body_(local_carry: tuple[torch.Tensor, ...], xs: torch.Tensor, local_out_ys: torch.Tensor):       
        for i in range(xs.shape[1]):
            local_carry, y = f(local_carry, xs[:, i])
            local_out_ys[:, i] = y
        return local_carry

    @partial(torch.compiler.disable, recursive = False)
    def do_uncompiled_loop():
        carry = init_carry
        for i in range(num_chunk):
            carry = unrolled_body_(carry, xs[:, i * unroll:][:, :unroll], out_ys[:, i * unroll:][:, :unroll])

    do_uncompiled_loop()
    return out_ys

def generic_scan_with_states(
    f: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[tuple[torch.Tensor, ...], torch.Tensor]], # f(s_t, x) -> (s_t+1, y)
    init: tuple[torch.Tensor, ...],
    xs: torch.Tensor,
    unroll: int = 1,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """ 
    Same logic that generic scan but return states.
    Only used for visualization purpose and should not be used with grad mode
    """
    num_chunk = math.ceil(xs.shape[1] / unroll)
    out_ys = torch.empty_like(xs)
    carry_out = torch.stack([torch.concat((x.unsqueeze(1).expand((xs.shape[0], 1, -1)), torch.empty_like(xs)), dim=1) for x in init], dim=0)
    def unrolled_body_(local_carry_out: tuple[torch.Tensor, ...], xs: torch.Tensor, local_out_ys: torch.Tensor):
        local_carry = local_carry_out[:, :, 0].unbind(0)
        for i in range(xs.shape[1]):
            local_carry, y = f(local_carry, xs[:, i])
            local_carry_out[:, :, i+1] = torch.stack(local_carry, 0)
            local_out_ys[:, i] = y
            
    @partial(torch.compiler.disable, recursive = False)
    def do_uncompiled_loop():
        for i in range(num_chunk):
            unrolled_body_(carry_out[:, :, i*unroll:, :][:, :, :unroll + 1, :], xs[:, i * unroll:][:, :unroll], out_ys[:, i * unroll:][:, :unroll])
    do_uncompiled_loop()
    return carry_out, out_ys

def A_law(x: torch.Tensor, a: float = 87.6):
    sign_x = torch.sign(x)
    abs_x = torch.abs(x)
    log_a = torch.log(a)
    y1 =  (a*abs_x)/(1 + log_a)
    y2 = (1 + torch.log(abs_x) + log_a)/(1 + log_a)
    y = torch.where(abs_x < 1/a, y1, y2)
    return sign_x*y

def inverse_A_law(y: torch.Tensor, a: float = 87.6):
    sign_y = torch.sign(y)
    abs_y = torch.abs(y)
    log_a_p1 = torch.log(a) + 1
    x1 = (abs_y*log_a_p1)/a
    x2 = torch.exp(-1 + abs_y*log_a_p1)/a
    x = torch.where(abs_y < 1/log_a_p1, x1, x2)    
    return sign_y*x


def adjust_neurons_for_parameter_budget(cfg: DictConfig):
    """
    Dynamically adjusts the number of neurons in a 2-layer network 
    to match a target parameter budget, accounting for active ablation flags.
    """
    # Only run if a target_params budget is specified in the config/CLI
    target_params = cfg.get('target_params', 0)
    if target_params == 0:
        return

    target_params = int(target_params)
    
    # 1. Extract structural constants
    I_data = cfg.l1.input_size
    # safely get num_classes (handles hydra interpolation)
    num_classes = cfg.l_out.n_neurons 
    C = cfg.l1.get('num_compartments', 1)
    
    # 2. Extract ablation flags
    P = 1 if cfg.l1.get('proximal_dendrite', True) else 0
    R_S2S = 1 if cfg.l1.get('use_recurrent', True) else 0
    R_D2D = 1 if cfg.l1.get('recurrent_dendrite', False) else 0
    R_S2Dself = 1 if cfg.l1.get('soma_to_dendrite_recurrence', False) else 0
    R_S2Dfull = 1 if cfg.l1.get('soma_to_dendrite_full_recurrence', False) else 0
    
    # 3. Formulate the Quadratic Equation (A*O^2 + B*O + C_const = Target)
    # W_factor determines if weights route to soma + dendrites, or just dendrites
    W_factor = P + C 
    
    # A_layer is the N^2 scaling factor per layer
    A_layer = R_S2S + (R_S2Dfull * C)
    
    # B_layer_0 is the linear scaling overhead per layer (states, taus, self-recurrences)
    # 2 + 4C accounts for u_reset, somatic bias, tau_u + d_reset, u_p, dendritic bias, tau_d, tau_t
    B_layer_0 = W_factor + 2 + (4 * C) + (R_D2D * (C ** 2)) + (R_S2Dself * C)
    
    # Combine for a 2-Layer network where L2 input = O
    A = (2 * A_layer) + W_factor
    B = (W_factor * I_data) + (2 * B_layer_0) + num_classes
    C_const = 2 * num_classes  # Readout layer biases and tau_u
    
    # 4. Solve the Quadratic Formula
    c_adj = C_const - target_params
    discriminant = (B ** 2) - (4 * A * c_adj)
    
    if discriminant < 0:
        raise ValueError(f"Cannot solve for {target_params} params. Imaginary roots. Budget too small for architecture.")
        
    # Calculate root and round to nearest integer
    O = (-B + math.sqrt(discriminant)) / (2 * A)
    n_neurons = int(round(O))
    
    # logging.info(f"--- DYNAMIC SIZING ENGAGED ---")
    # logging.info(f"Target Params : {target_params}")
    # logging.info(f"Calculated    : {n_neurons} neurons per layer")
    
    # 5. Override the config objects dynamically
    cfg.l1.n_neurons = n_neurons
    cfg.l2.input_size = n_neurons
    cfg.l2.n_neurons = n_neurons
    cfg.l_out.input_size = n_neurons