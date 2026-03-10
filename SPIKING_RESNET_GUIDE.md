# Spiking ResNet Integration Guide

This document explains how to use the new Spiking ResNet architecture in your framework.

## Overview

The Spiking ResNet extends your framework to support residual neural networks with configurable spiking neuron models. Just like your existing 2-layer RSNN, the ResNet supports:

- **Multiple neuron types**: LIF, SE-adLIF, EF-adLIF, MCLIF, MCadLIF
- **Multiple stages**: Configurable number of residual stages
- **Configurable depth**: Different number of blocks per stage
- **Configurable width**: Different channel widths per stage
- **Skip connections**: Automatic handling of dimension mismatches

## Files Added/Modified

### New Files
- `models/resnet.py` - Core Spiking ResNet and SpikingResidualBlock implementations
- `config/experiment/SHD_ResNet_LIF_S.yaml` - Small ResNet with LIF example
- `config/experiment/SHD_ResNet_LIF_M.yaml` - Medium ResNet with LIF example
- `config/experiment/SHD_ResNet_SEadLIF_S.yaml` - Small ResNet with SE-adLIF example
- `config/experiment/SHD_ResNet_EFadLIF_S.yaml` - Small ResNet with EF-adLIF example

### Modified Files
- `models/pl_module.py` - Added `SpikingResNetSNN` Lightning module
- `run.py` - Added model type selection logic

## Usage

### Basic Configuration

To use Spiking ResNet, set `model_type: resnet` in your experiment config and configure the ResNet parameters:

```yaml
model_type: resnet

resnet:
  neuron_type: lif  # or se_adlif, ef_adlif, mclif, mcalif
  input_size: 140
  n_neurons: 20  # number of output classes
  
  # Architecture: 3 stages with [2, 2, 2] blocks and [64, 128, 256] channels
  num_stages: 3
  blocks_per_stage: [2, 2, 2]
  channels_per_stage: [64, 128, 256]
  
  # Neuron parameters (same as your existing models)
  tau_u_range: [5, 25]
  use_recurrent: True
  alpha: 5.0
  c: 0.1
  dt: 1.0
  train_tau: interpolation
```

### Configuration Parameters

#### ResNet Structure
- `num_stages`: Number of residual stages (default: 3)
- `blocks_per_stage`: List of block counts per stage, e.g., `[2, 2, 2]`
- `channels_per_stage`: List of output channels per stage, e.g., `[64, 128, 256]`

#### Neuron Selection
- `neuron_type`: One of:
  - `lif` - Leaky Integrate-and-Fire
  - `se_adlif` - SE-adLIF with synaptic weight adaptation
  - `ef_adlif` - EF-adLIF with firing rate adaptation
  - `mclif` - Multi-Compartment LIF
  - `mcalif` - Multi-Compartment adLIF

#### Neuron Parameters
All parameters from your existing neuron models are supported:
- `tau_u_range`: Memory time constant range
- `tau_w_range`: Adaptation time constant range (for adLIF models)
- `use_recurrent`: Enable recurrent connections
- `train_tau`: Tau training method
- `a_range`, `b_range`: Adaptation parameter ranges (for adLIF)
- `alpha`, `c`: Surrogate gradient parameters

### Example Configurations

#### Small ResNet (2 stages, 2 blocks each)
```yaml
num_stages: 2
blocks_per_stage: [2, 2]
channels_per_stage: [64, 128]
```

#### Medium ResNet (3 stages)
```yaml
num_stages: 3
blocks_per_stage: [2, 2, 2]
channels_per_stage: [64, 128, 256]
```

#### Large ResNet (4 stages with varying depths)
```yaml
num_stages: 4
blocks_per_stage: [2, 2, 3, 2]
channels_per_stage: [64, 128, 256, 512]
```

### Running Your First ResNet Experiment

1. **Using a pre-made config:**
```bash
python run.py experiment=SHD_ResNet_LIF_S
```

2. **Creating a custom config:**
Create `config/experiment/CIFAR_ResNet_SEadLIF.yaml`:
```yaml
# @package _global_
defaults:
  - /dataset: cifar10dvs

exp_name: CIFAR_ResNet_SEadLIF
model_type: resnet

resnet:
  neuron_type: se_adlif
  input_size: 32  # CIFAR-10 DVS spatial resolution
  n_neurons: 10   # number of classes
  num_stages: 3
  blocks_per_stage: [2, 2, 2]
  channels_per_stage: [64, 128, 256]
  tau_u_range: [5, 25]
  tau_w_range: [10, 100]
  use_recurrent: True
  alpha: 5.0
  c: 0.1
  dt: 1.0

dropout: 0.1
n_epochs: 300
batch_size: 128
lr: 0.001
factor: 0.5
patience: 20
tracking_metric: val_acc_epoch
tracking_mode: max
```

Then run:
```bash
python run.py experiment=CIFAR_ResNet_SEadLIF
```

## Architecture Details

### Residual Block Structure

Each residual block contains:
1. **First neuron layer**: Processes input, outputs to second layer
2. **Second neuron layer**: Processes first layer output
3. **Skip connection**: Adds the input (with dimension adjustment if needed)

Skip connections automatically handle:
- **Channel mismatch**: Uses a learned linear transformation
- **Stride**: Uses adaptive average pooling if stride > 1

### Full Network Structure

```
Input (batch, time, input_size)
    ↓
Input Neuron Layer → (batch, time, channels_per_stage[0])
    ↓
Stage 1 (blocks_per_stage[0] residual blocks)
    ↓
Stage 2 (blocks_per_stage[1] residual blocks) [stride=2 in first block]
    ↓
Stage 3 (blocks_per_stage[2] residual blocks) [stride=2 in first block]
    ↓
Output Layer (LI neuron) → (batch, time, num_classes)
```

## Comparison with Your Existing 2-Layer RSNN

| Feature | 2-Layer RSNN | Spiking ResNet |
|---------|-------------|-------------|
| Configurable neurons | ✓ | ✓ |
| Network depth | Fixed (2 layers) | Configurable (1-4+ stages) |
| Width progression | All layers same size | Channels increase per stage |
| Skip connections | None | Yes |
| Capacity | Low | Medium to High |
| Suitable for | Simpler tasks | Complex temporal patterns |

## Training Tips

1. **Learning Rate**: Start with 0.001, adjust based on dataset complexity
2. **Dropout**: Increase if overfitting (recommend 0.1-0.2 for ResNets)
3. **Stages**: More stages = more capacity but also more parameters:
   - Small datasets: 2-3 stages
   - Medium datasets: 3 stages
   - Large datasets: 3-4 stages
4. **Block Depth**: Deeper blocks (more blocks per stage) help with complex patterns
5. **Neuron Type**: 
   - LIF: Fastest, good baseline
   - SE/EF-adLIF: Better temporal dynamics, slightly slower

## Known Considerations

1. **Skip Connection Dimensions**: Automatically handled, but ensure your `channels_per_stage` progression makes sense
2. **Memory**: Larger ResNets (4+ stages with high channels) require more VRAM
3. **Computation**: More blocks = more computation, plan accordingly for your hardware

## Troubleshooting

**Issue: CUDA out of memory**
- Solution: Reduce `channels_per_stage` or `blocks_per_stage`, or use smaller batch size

**Issue: Loss doesn't decrease**
- Solution: Try a smaller learning rate (0.0001), adjust `alpha` and `c` for surrogate gradient

**Issue: Model trains slowly**
- Solution: Use LIF instead of SE/EF-adLIF, or reduce number of stages

## Next Steps

You can now:
- Experiment with different architectures using the provided configs
- Create new experiment configs for different datasets
- Fine-tune hyperparameters for your specific tasks
- Combine ResNets with your existing compression and evaluation pipelines
