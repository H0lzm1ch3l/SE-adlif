import torch
import pytorch_lightning as pl
import json
from models.pl_module import MLPSNN
from torch.nn import functional as F

# Load the checkpoint
model_path = "/raid/home/michael.siegl/projects/SE-adlif/results/hydra/2026-03-21/20-17-44/ckpt/epoch=39-step=11760.ckpt"
checkpoint = torch.load(model_path, map_location='cpu')
hparams = checkpoint['hyper_parameters']

# Instantiate the model with hparams
model = MLPSNN(**hparams)
model.load_state_dict(checkpoint['state_dict'])
model.to('cpu')
print("Model loaded")
print("Input size:", hparams['cfg']['l1']['input_size'])

# Load a sample from SSC
# For demo, create mock events
time_steps = 100
input_size = hparams['cfg']['l1']['input_size']
events = torch.randn(time_steps, input_size)
target = 5
block_idx = torch.ones((time_steps,), dtype=torch.int64)
events = events.to(model.device)

# Run the model and collect states
model.eval()
with torch.no_grad():
    initial_states = model.l1.initial_state(1, events.device)
    print("len initial_states:", len(initial_states))
    s1 = initial_states
    print("len(s1):", len(s1))
    print("type s1:", type(s1))
    print("s1[0] shape:", s1[0].shape)
    states_list = []
    out_sequence = []
    for t, x_t in enumerate(events.unbind(0)):
        x_t = x_t.unsqueeze(0)  # Add batch dimension
        decay_u = model.l1.tau_u_trainer.get_decay()
        decay_d = model.l1.tau_d_trainer.get_decay()
        decay_t = model.l1.tau_t_trainer.get_decay()
        current = F.linear(x_t, model.l1.weight, model.l1.bias)
        new_states, z_t = model.l1.step(model.l1.recurrent, decay_u, decay_d, decay_t, model.l1.s_thr, model.l1.d_thr, model.l1.u0, model.l1.d0, s1, current)
        s1 = new_states
        out_sequence.append(z_t.squeeze(0))
        states_list.append(new_states)

out = torch.stack(out_sequence, dim=0)
num_compartments = hparams['cfg']['l1']['num_compartments']
out_features = hparams['cfg']['l1']['n_neurons']

# Format data for JSON
spikes = []
dendrites = []
for t in range(len(states_list)):
    u, z, d, t_state, p = states_list[t]
    # Spikes: where z > 0
    z_flat = z.squeeze(0)  # Remove batch dim
    spike_indices = torch.nonzero(z_flat > 0).squeeze(-1)
    for idx in spike_indices:
        spikes.append({"t": t, "neuron": int(idx), "type": "soma"})
    
    # Dendrites: d values
    d_flat = d.squeeze(0)  # (out_features, num_compartments)
    for neuron in range(out_features):
        for comp in range(num_compartments):
            dendrites.append({
                "t": t,
                "neuron": neuron,
                "compartment": comp,
                "value": float(d_flat[neuron, comp])
            })

data = {
    "spikes": spikes,
    "dendrites": dendrites,
    "time_steps": len(states_list),
    "num_neurons": out_features,
    "num_compartments": num_compartments,
    "target": int(target)
}

# Save to JSON
with open("neuron_viz_data.json", "w") as f:
    json.dump(data, f)

print("Data saved to neuron_viz_data.json")