import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig, OmegaConf
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
        self.use_sparse = cfg.get('use_sparse', True)
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


class DHSSRNLayer(nn.Module):
    """Dendritic Heterogeneity Spiking Recurrent Neural Network (DH-SRNN) layer.
    
    Incorporates recurrent feedback from previous timestep spikes combined with
    feedforward input through dendritic branches.
    """
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_size
        self.output_dim = cfg.get('n_neurons', cfg.get('output_dim'))
        if self.output_dim is None:
            raise ValueError("DHSRNN layer requires n_neurons or output_dim")

        self.vth = cfg.get('vth', 0.5)
        self.dt = cfg.get('dt', 1)
        self.branch = cfg.get('branch', 4)
        self.use_sparse = cfg.get('use_sparse', True)
        self.test_sparsity = cfg.get('test_sparsity', False)
        self.sparsity = cfg.get('sparsity', 0.5)
        self.mask_share = cfg.get('mask_share', 1)

        # Input + recurrent dimension for dense layer
        self.total_dim = self.input_dim + self.output_dim
        self.pad = ((self.total_dim) // self.branch * self.branch + self.branch - self.total_dim) % self.branch
        self.dense = nn.Linear(self.total_dim + self.pad, self.output_dim * self.branch, bias=True)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, self.branch))
        self.register_buffer(
            'mask', torch.zeros(self.output_dim * self.branch, self.total_dim + self.pad)
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
        input_size = self.total_dim + self.pad
        self.mask.zero_()
        for i in range(self.output_dim // max(1, self.mask_share)):
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
        # Concatenate feedforward input with recurrent spikes from previous timestep
        padding = torch.zeros(
            input_spike.size(0), self.pad, device=input_spike.device, dtype=input_spike.dtype
        )
        k_input = torch.cat((input_spike.float(), self.spike, padding), dim=1)
        beta = torch.sigmoid(self.tau_n)
        print(k_input.shape)
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


class BidirectionalDHSSNN(nn.Module):
    """Bidirectional wrapper for DH-SNN/DH-SRNN layers.
    
    Processes sequence forward and backward, concatenating outputs at each timestep.
    """
    def __init__(self, forward_layer, backward_layer=None):
        super().__init__()
        self.forward_layer = forward_layer
        # Use same layer config for backward if not specified
        self.backward_layer = backward_layer

    def initial_state(self, batch_size, device):
        self.forward_layer.initial_state(batch_size, device)
        if self.backward_layer is not None:
            self.backward_layer.initial_state(batch_size, device)
        return None

    def forward(self, input_spike, state=None, reverse=False):
        if reverse:
            # Reverse temporal order for backward pass
            input_reversed = torch.flip(input_spike, dims=[1])
            out_fwd, _ = self.forward_layer(input_reversed, None)
            # Flip back to original order
            out_fwd = torch.flip(out_fwd, dims=[1])
            return out_fwd, state
        else:
            out, _ = self.forward_layer(input_spike, None)
            return out, state

    def apply_mask(self):
        if hasattr(self.forward_layer, 'apply_mask'):
            self.forward_layer.apply_mask()
        if self.backward_layer is not None and hasattr(self.backward_layer, 'apply_mask'):
            self.backward_layer.apply_mask()


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


def normalize_dhsnn_layers(cfg):
    """Normalize hidden layer config for DHSNN.
    
    Supports:
    - hidden_layers: list of layer configs (new style)
    - l1/l2 + two_layers: legacy config (backward compatible)
    - layer_type: 'snn' or 'srnn' to select dense vs recurrent
    """
    if cfg.get('hidden_layers') is not None:
        return cfg.hidden_layers
    
    hidden_layers = [cfg.l1]
    if cfg.get('two_layers', False):
        hidden_layers.append(cfg.l2)
    return hidden_layers


def count_dhsnn_params(layer_cfg, is_recurrent=False, is_bidirectional=False):
    """Calculate parameter count for a single DHSNN/DH-SRNN layer.
    
    NOTE: This calculates the ACTUAL model parameters after all factors are applied.
    The solver uses this to find neuron counts that achieve target_params.
    
    Args:
        layer_cfg: DictConfig with input_size, n_neurons, branch, use_sparse
        is_recurrent: If True, layer is DH-SRNN (input + output as input)
        is_bidirectional: If True, layer is processed in both directions
    
    Returns:
        int: Total actual parameter count
    """
    input_dim = layer_cfg.input_size
    output_dim = layer_cfg.get('n_neurons', layer_cfg.get('output_dim', 64))
    branch = layer_cfg.get('branch', 4)
    use_sparse = layer_cfg.get('use_sparse', False)
    sparsity = layer_cfg.get('sparsity', 0.5)
    
    # Calculate padding for branch alignment
    if is_recurrent:
        total_dim = input_dim + output_dim
    else:
        total_dim = input_dim
    pad = ((total_dim) // branch * branch + branch - total_dim) % branch
    
    # Dense layer: (input + pad) x (output x branch) weights + (output x branch) biases
    dense_in = total_dim + pad
    dense_out = output_dim * branch
    dense_params = dense_in * dense_out + dense_out  # weights + biases
    
    # tau_m: output_dim parameters
    tau_m_params = output_dim
    
    # tau_n: output_dim x branch parameters
    tau_n_params = output_dim * branch
    
    total = dense_params + tau_m_params + tau_n_params
    
    # Apply sparsity reduction to dense layer parameters
    # Sparse connections reduce the effective weight matrix size
    if use_sparse:
        total = dense_params * (1 - sparsity) + tau_m_params + tau_n_params
    
    # Bidirectional: factor in additional processing for parameter estimation
    # This is used by the solver to find correct neuron count
    if is_bidirectional:
        total = int(total * 1.5)
    
    return total


def solve_neurons_for_params(target_params, input_dim, branch, is_recurrent=False, is_bidirectional=False, output_dim_hint=None):
    """Solve for neuron count that achieves target parameter count.
    
    NOTE: Only bidirectionality is used for compensation (1.5x factor).
    Sparse is NOT compensated - it reduces actual params but doesn't affect neuron count.
    
    Args:
        target_params: Target parameter count
        input_dim: Input feature dimension
        branch: Number of dendritic branches
        is_recurrent: If True, use DH-SRNN formula (input + output as input)
        is_bidirectional: If True, layer is processed bidirectionally (used for compensation)
        output_dim_hint: Optional hint for output dimension (used for recurrent)
    
    Returns:
        int: Neuron count that achieves target_params
    """
    # For recurrent, we need to solve iteratively since output affects input
    if is_recurrent:
        # Iterative solution for recurrent layers
        n = output_dim_hint or 64  # start with hint or default
        for _ in range(100):
            total_dim = input_dim + n
            pad = ((total_dim) // branch * branch + branch - total_dim) % branch
            # Dense params calculation
            dense_in = total_dim + pad
            dense_out = n * branch
            dense_params = dense_in * dense_out + dense_out
            
            # tau_m + tau_n
            tau_params = n + n * branch
            
            # Full dense params (sparse NOT used for compensation)
            total_layer = dense_params + tau_params
            
            # Bidirectional: factor in for compensation (sparse is NOT compensated)
            if is_bidirectional:
                total_layer *= 1.5
            
            predicted = n * total_layer
            if abs(predicted - target_params) < target_params * 0.001:
                break
            # Adjust: n = target / coeff
            n = max(1, int(target_params / total_layer))
        return max(1, n)
    
    # Non-recurrent: closed-form solution
    pad = ((input_dim) // branch * branch + branch - input_dim) % branch
    # Dense params calculation (sparse NOT used for compensation)
    dense_in = input_dim + pad
    dense_out = branch  # Will be multiplied by n
    coeff = branch * (input_dim + pad + 2) + 1
    
    # Bidirectional: factor in for compensation (sparse is NOT compensated)
    if is_bidirectional:
        coeff *= 1.5
    
    n = max(1, int(target_params / coeff))
    return n


def count_readout_params(layer_cfg):
    """Calculate parameter count for readout layer."""
    input_dim = layer_cfg.input_size
    output_dim = layer_cfg.get('n_neurons', layer_cfg.get('output_dim', 10))
    
    # Dense: input x output weights + output biases
    dense_params = input_dim * output_dim + output_dim
    # tau_m: output_dim parameters
    tau_m_params = output_dim
    
    return dense_params + tau_m_params


def calculate_total_params(cfg):
    """Calculate total parameter count for DHSNN model from config.
    
    NOTE: Bidirectionality is factored (1.5x) for the parameter estimate.
    Sparse is NOT factored in compensation - actual params will be lower when sparse is enabled.
    
    Args:
        cfg: DictConfig with hidden_layers, l_out, use_recurrent, use_bidirectional
    
    Returns:
        int: Total parameter count
    """
    hidden_layers = normalize_dhsnn_layers(cfg)
    use_recurrent = cfg.get('use_recurrent', False)
    use_bidirectional = cfg.get('use_bidirectional', False)
    
    total = 0
    for layer_cfg in hidden_layers:
        layer_type = layer_cfg.get('layer_type', 'snn')
        is_recurrent = (layer_type == 'srnn') or use_recurrent
        total += count_dhsnn_params(layer_cfg, is_recurrent, use_bidirectional)
    
    # Readout layer
    total += count_readout_params(cfg.l_out)
    
    return total


def set_neurons_for_target_params(target_params, hidden_layers_cfg, use_recurrent=False, use_bidirectional=False):
    """Set neuron counts in hidden_layers to achieve target parameter count.
    
    NOTE: Only bidirectionality is compensated (1.5x factor).
    Sparse is NOT compensated - it reduces actual params but doesn't affect neuron count.
    
    Args:
        target_params: Target parameter count (e.g., 1600000 for 1.6M)
        hidden_layers_cfg: List of layer configs (will be modified in place)
        use_recurrent: Whether using recurrent layers
        use_bidirectional: Whether using bidirectional processing
    
    Returns:
        tuple: (modified hidden_layers_cfg, actual total params)
    """
    # Calculate readout params first (fixed)
    if not hidden_layers_cfg:
        raise ValueError("Need at least one hidden layer config")
    
    # Handle both dict and OmegaConf
    def get_n(cfg, key, default):
        if hasattr(cfg, 'get'):
            return cfg.get(key, default)
        return cfg.get(key, default) if isinstance(cfg, dict) else default
    
    last_output = get_n(hidden_layers_cfg[-1], 'n_neurons', 64)
    last_n_neurons = get_n(hidden_layers_cfg[-1], 'n_neurons', 10)
    readout_params = count_readout_params({
        'input_size': last_output,
        'n_neurons': last_n_neurons
    })
    
    hidden_budget = target_params - readout_params
    if hidden_budget <= 0:
        raise ValueError(f"Target params {target_params} too small. Need at least {readout_params} for readout.")
    
    # Distribute budget across layers (equal distribution)
    num_layers = len(hidden_layers_cfg)
    budget_per_layer = hidden_budget / num_layers
    
    # Solve for neurons in each layer
    prev_output = hidden_layers_cfg[0].input_size
    for i, layer_cfg in enumerate(hidden_layers_cfg):
        input_dim = layer_cfg.input_size
        branch = layer_cfg.get('branch', 4)
        
        # Use previous layer's output as input hint for recurrent
        n = solve_neurons_for_params(
            budget_per_layer, 
            input_dim, 
            branch, 
            is_recurrent=use_recurrent,
            is_bidirectional=use_bidirectional,
            output_dim_hint=prev_output
        )
        
        layer_cfg['n_neurons'] = n
        if 'output_dim' in layer_cfg:
            layer_cfg['output_dim'] = n
        prev_output = n
    
    # Verify total (pass bidirectional info to calculate_total_params)
    total = calculate_total_params(OmegaConf.create({
        'hidden_layers': hidden_layers_cfg,
        'l_out': {'input_size': prev_output, 'n_neurons': hidden_layers_cfg[-1].get('n_neurons', 10)},
        'use_recurrent': use_recurrent,
        'use_bidirectional': use_bidirectional
    }))
    
    return hidden_layers_cfg, total


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
        
        # DH-SRNN specific options
        self.use_recurrent = cfg.get('use_recurrent', False)
        self.use_bidirectional = cfg.get('use_bidirectional', False)
        self.use_sparse = cfg.get('use_sparse', True)

        # Build hidden layers
        self.hidden_layers_cfg = normalize_dhsnn_layers(cfg)
        self.hidden_layers = nn.ModuleList()
        self.num_hidden_layers = len(self.hidden_layers_cfg)
        
        for layer_cfg in self.hidden_layers_cfg:
            layer_type = layer_cfg.get('layer_type', 'snn')
            if layer_type == 'srnn' or self.use_recurrent:
                self.hidden_layers.append(DHSSRNLayer(layer_cfg))
            else:
                self.hidden_layers.append(DHSSNNLayer(layer_cfg))
        
        self.out_layer = DHReadoutLayer(cfg.l_out)

        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Initialize states for all hidden layers
        for layer in self.hidden_layers:
            layer.initial_state(batch_size, device)
        self.out_layer.initial_state(batch_size, device)

        out_sequence = torch.zeros(
            (batch_size, inputs.shape[1], self.output_size), device=device
        )
        sparsity_sequences = torch.zeros(
            (inputs.shape[1], self.num_hidden_layers), device=device
        )
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))

        if self.use_bidirectional:
            # Forward pass
            out_fwd, sparsity_fwd = self._forward_pass(
                inputs, batch_size, device, single_step_prediction_limit, reverse=False
            )
            # Backward pass
            out_bwd, sparsity_bwd = self._forward_pass(
                inputs, batch_size, device, single_step_prediction_limit, reverse=True
            )
            # Concatenate forward and backward at each timestep (NOT add)
            out_sequence = torch.cat([out_fwd, out_bwd], dim=-1)
            sparsity_sequences = (sparsity_fwd + sparsity_bwd) / 2
        else:
            out_sequence, sparsity_sequences = self._forward_pass(
                inputs, batch_size, device, single_step_prediction_limit, reverse=False
            )

        self.sparsity_sequences = sparsity_sequences.mean(dim=0)
        return out_sequence

    def _forward_pass(self, inputs, batch_size, device, single_step_prediction_limit, reverse=False):
        """Internal forward pass supporting forward/backward directions."""
        if reverse:
            inputs = torch.flip(inputs, dims=[1])
        
        seq_len = inputs.shape[1]
        out_features = self.out_layer.output_dim
        out_sequence = torch.zeros(
            (batch_size, seq_len, out_features), device=device
        )
        sparsity_sequences = torch.zeros(
            (seq_len, self.num_hidden_layers), device=device
        )
        
        for t, x_t in enumerate(inputs.unbind(1)):
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()
            
            out = x_t
            for layer_idx, layer in enumerate(self.hidden_layers):
                out, _ = layer(out, None)
                sparsity_sequences[t, layer_idx] = out.mean()
                out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            
            out, _ = self.out_layer(out, None)
            out_sequence[:, t] = out

        if reverse:
            out_sequence = torch.flip(out_sequence, dims=[1])
            sparsity_sequences = torch.flip(sparsity_sequences, dims=[0])
        
        return out_sequence, sparsity_sequences

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        for layer in self.hidden_layers:
            if hasattr(layer, 'apply_mask'):
                layer.apply_mask()

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
