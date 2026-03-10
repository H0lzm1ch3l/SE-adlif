"""
Spiking ResNet implementation with configurable neuron models
Compatible with LIF, SEAdLIF, EFAdLIF, and other neuron models
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.parameter import Parameter
from omegaconf import DictConfig, OmegaConf

from models.helpers import init_micheli_normal
from models.lif import LIF
from models.alif import EFAdLIF, SEAdLIF
from models.mclif import MCLIF
from models.mcalif import MCAdLIF
from models.li import LI


# Neuron cell map
neuron_map = {
    "lif": LIF,
    "mclif": MCLIF,
    "mcalif": MCAdLIF,
    "se_adlif": SEAdLIF,
    "ef_adlif": EFAdLIF,
}

class SpikingResidualBlock(Module):
    """
    A residual block using spiking neurons for SNNs.
    Supports skip connections with optional dimension adjustment.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        neuron_class,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Create neuron layer with explicit config (already has spatial info)
        self.neuron = neuron_class(cfg)
        
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            device=device,
            dtype=dtype,
        )
        init_micheli_normal(self.conv2d.weight, factor=self.kernel_size**2)
        torch.nn.init.zeros_(self.conv2d.bias)
        
        # Primary path batch norm
        self.batchnorm = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)

        # Skip connection adjustment (if needed)
        self.skip_adjustment = None
        self.skip_conv = None
        self.skip_bn = None
        if stride > 1 or in_channels != out_channels:
            if in_channels != out_channels:
                # Use a 1x1 conv for channel adjustment and a separate batch-norm
                self.skip_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=True,
                    device=device,
                    dtype=dtype,
                )
                init_micheli_normal(self.skip_conv.weight, factor=1)
                self.skip_bn = nn.BatchNorm2d(out_channels, device=device, dtype=dtype)
            else:
                # Just stride without channel change (downsample spatially)
                if stride > 1:
                    self.skip_conv = nn.MaxPool2d(kernel_size=stride, stride=stride)
                    # no batch-norm needed for simple downsampling
                else:
                    self.skip_conv = None
    
    def forward(
        self,
        input_tensor: Tensor,
        states: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for a single timestep.
        
        Args:
            input_tensor: (batch_size, in_channels, height, width)
            states: state for the neuron
        
        Returns:
            output: (batch_size, out_channels, height, width)
            new_states: updated state
        """
        # Neuron forward pass
        
        conv_result = self.conv2d(input_tensor)
        conv_result = self.batchnorm(conv_result)
        
        shape = conv_result.shape
        out, state_new = self.neuron(conv_result.flatten(1, 3), states)
        out = out.view(shape[0], self.out_channels, shape[2], shape[3])
        
        # Skip connection
        skip = input_tensor
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
            if self.skip_bn is not None:
                skip = self.skip_bn(skip)
        
        # Add skip connection
        output = out + skip
        
        return output, state_new
    
    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tensor:
        """Initialize state for the neuron"""
        return self.neuron.initial_state(batch_size, device)
    
    def apply_parameter_constraints(self):
        """Apply parameter constraints"""
        self.neuron.apply_parameter_constraints()


class SpikingResNet(Module):
    """
    Spiking Residual Network with configurable neuron models and depth.
    Mimics standard ResNet architecture but with spiking neurons.
    
    Supports multiple stages of residual blocks, each with potentially
    different depths and channel widths.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Get resnet config from Hydra
        resnet_cfg = cfg.resnet
        neuron_cfg = cfg.neuron_cfg
        
        # Initial spatial dimensions
        self.height = resnet_cfg.get('input_height', 48)
        self.width = resnet_cfg.get('input_width', 48)
        
        # Initial convolution
        initial_conv_cfg = resnet_cfg.initial_conv
        self.initial_conv = nn.Conv2d(
            in_channels=initial_conv_cfg.in_channels,
            out_channels=initial_conv_cfg.out_channels,
            kernel_size=initial_conv_cfg.kernel_size,
            stride=initial_conv_cfg.stride,
            padding=initial_conv_cfg.padding,
            bias=initial_conv_cfg.bias,
            device=device,
            dtype=dtype,
        )
        
        # Calculate spatial dimensions after initial conv
        k = initial_conv_cfg.kernel_size
        s = initial_conv_cfg.stride
        p = initial_conv_cfg.padding
        self.height = (self.height + 2*p - k) // s + 1
        self.width = (self.width + 2*p - k) // s + 1
        
        # Parse stages from config
        stages_cfg_list = resnet_cfg.stages
        self.stages = nn.ModuleList()
        
        in_channels = initial_conv_cfg.out_channels
        current_height = self.height
        current_width = self.width
        
        for stage_cfg in stages_cfg_list:
            out_channels = stage_cfg.out_channels
            num_blocks = stage_cfg.num_blocks
            stride = stage_cfg.stride
            kernel_size = stage_cfg.get('kernel_size', 3)
            padding = stage_cfg.get('padding', 1)
            neuron_cell_name = stage_cfg.neuron_cell
            neuron_class = neuron_map[neuron_cell_name]
            
            stage_blocks = nn.ModuleList()
            
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                # Update spatial dimensions after first block in stage (which has the stride)
                if block_stride > 1:
                    current_height = (current_height + 2*padding - kernel_size) // stride + 1
                    current_width = (current_width + 2*padding - kernel_size) // stride + 1
                block_in_channels = in_channels if block_idx == 0 else out_channels
                
                # Create neuron config with spatial dimensions
                block_cfg = OmegaConf.create(OmegaConf.to_container(neuron_cfg))
                block_cfg.input_size = block_in_channels
                block_cfg.n_neurons = out_channels * current_height * current_width
                block_cfg.width = current_width
                block_cfg.height = current_height
                block_cfg.kernel_size = kernel_size
                block_cfg.padding = padding
                block_cfg.stride = block_stride
                
                block = SpikingResidualBlock(
                    block_cfg,
                    neuron_class,
                    block_in_channels,
                    out_channels,
                    stride=block_stride,
                    kernel_size=kernel_size,
                    padding=padding,
                    device=device,
                    dtype=dtype,
                )
                stage_blocks.append(block)
            
            self.stages.append(stage_blocks)
            in_channels = out_channels
        
        # Global average pooling
        self.global_avg_pool = resnet_cfg.get('global_avg_pool', True)
        self.final_channels = in_channels
        self.final_height = current_height
        self.final_width = current_width
        
        # Output layer configuration
        output_cfg = cfg.l_out
        # Set input size to flattened spatial features after global avg pool
        if self.global_avg_pool:
            output_cfg.input_size = self.final_channels
        else:
            output_cfg.input_size = self.final_channels * self.final_height * self.final_width
        
        self.output_layer = LI(output_cfg, device=device, dtype=dtype)
        
        self.save_config(cfg)
    
    def save_config(self, cfg: DictConfig):
        """Save configuration for reference"""
        self.register_buffer("_cfg_saved", torch.tensor(1.0))
    
    def initial_state(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple:
        """
        Initialize states for all layers.
        Returns a nested tuple structure matching the network topology.
        """
        stage_states = []
        for stage in self.stages:
            stage_block_states = []
            for block in stage:
                block_state = block.initial_state(batch_size, device)
                stage_block_states.append(block_state)
            stage_states.append(stage_block_states)
        
        output_state = self.output_layer.initial_state(batch_size, device)
        
        return (stage_states, output_state)
    
    def forward(
        self, input_tensor: Tensor, states: Tuple
    ) -> Tuple[Tensor, Tuple]:
        """
        Forward pass for a single timestep.
        
        Args:
            input_tensor: (batch_size, channels, height, width)
            states: nested tuple of states for all layers
        
        Returns:
            output: (batch_size, num_classes)
            new_states: nested tuple of updated states
        """
        stage_states, output_state = states
        
        # Apply initial conv
        x = self.initial_conv(input_tensor)  # (batch, channels, height, width)
        
        # Residual stages
        stage_states_new = []
        for stage_idx, stage in enumerate(self.stages):
            stage_block_states_new = []
            for block_idx, block in enumerate(stage):
                block_state = stage_states[stage_idx][block_idx]
                x, block_state_new = block(x, block_state)
                stage_block_states_new.append(block_state_new)
            stage_states_new.append(stage_block_states_new)
        
        # Global average pooling
        if self.global_avg_pool:
            # (batch, channels, height, width) -> (batch, channels)
            x = x.mean(dim=[2, 3])
        else:
            # Flatten spatial dimensions
            x = x.view(x.shape[0], -1)
        
        # Output layer
        out, output_state_new = self.output_layer(x, output_state)
        
        return out, (stage_states_new, output_state_new)
    
    def layer_forward(self, inputs: torch.Tensor) -> Tensor:
        """
        Process a full sequence through the network.
        
        Args:
            inputs: (batch_size, time_steps, in_features)
        
        Returns:
            outputs: (batch_size, time_steps, out_features)
        """
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        device = inputs.device
        
        states = self.initial_state(batch_size, device)
        outputs = []
        
        for t in range(time_steps):
            out, states = self.forward(inputs[:, t], states)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)
    
    @torch.no_grad()
    def layer_forward_with_states(self, inputs: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """
        Process a full sequence and return final states.
        
        Args:
            inputs: (batch_size, time_steps, in_features)
        
        Returns:
            states: final state tuple
            outputs: (batch_size, time_steps, out_features)
        """
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        device = inputs.device
        
        states = self.initial_state(batch_size, device)
        outputs = []
        
        for t in range(time_steps):
            out, states = self.forward(inputs[:, t], states)
            outputs.append(out)
        
        return states, torch.stack(outputs, dim=1)
    
    def apply_parameter_constraints(self):
        """Apply parameter constraints to all layers"""
        for stage in self.stages:
            for block in stage:
                block.apply_parameter_constraints()
        
        self.output_layer.apply_parameter_constraints()
