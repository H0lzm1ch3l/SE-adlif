import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

# Load the data
with open('neuron_viz_data.json', 'r') as f:
    data = json.load(f)

# Create tb_logs directory if it doesn't exist
os.makedirs('tb_logs', exist_ok=True)

writer = SummaryWriter('tb_logs')

num_neurons = data['num_neurons']
num_compartments = data['num_compartments']
time_steps = data['time_steps']

# Create images for each time step
for t in range(min(50, time_steps)):  # Limit to first 50 time steps for demo
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get data for this time step
    dendrites_t = [d for d in data['dendrites'] if d['t'] == t]
    spikes_t = [s for s in data['spikes'] if s['t'] == t]
    spike_neurons = set(s['neuron'] for s in spikes_t)
    
    # Plot each neuron
    for neuron in range(min(20, num_neurons)):  # Limit to first 20 neurons for readability
        y_pos = neuron
        
        # Plot dendrites
        dend_values = []
        for comp in range(num_compartments):
            dend_data = next((d for d in dendrites_t if d['neuron'] == neuron and d['compartment'] == comp), None)
            value = dend_data['value'] if dend_data else 0
            dend_values.append(value)
            
            # Color based on value (blue for negative, red for positive)
            color = 'red' if value > 0 else 'blue'
            alpha = min(abs(value) * 2, 1)  # Scale alpha
            ax.barh(y_pos, 0.3, left=comp*0.35, height=0.8, color=color, alpha=alpha)
        
        # Plot soma
        spike_color = 'red' if neuron in spike_neurons else 'lightgray'
        ax.scatter(num_compartments*0.35 + 0.5, y_pos, s=100, c=spike_color, edgecolors='black')
    
    ax.set_xlim(0, num_compartments*0.35 + 1)
    ax.set_ylim(-0.5, min(20, num_neurons) - 0.5)
    ax.set_xlabel('Compartment / Soma')
    ax.set_ylabel('Neuron')
    ax.set_title(f'Neuron Activity at Time Step {t}')
    ax.grid(True, alpha=0.3)
    
    # Save figure to tensorboard
    writer.add_figure(f'Neuron Activity/Time_{t:03d}', fig, t)
    plt.close(fig)

writer.close()
print("Images logged to TensorBoard")