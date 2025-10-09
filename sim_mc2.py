import numpy as np
import matplotlib.pyplot as plt
from neuron_models.adlif import adLIF
from neuron_models.lif import LIF
from neuron_models.dendrite import dendrite, sech
from neuron_models.mclif import MC

# Constants (scaled to fixed-point representation)
dt = 0.1  # ms per timestep
timesteps = 400  # 40 ms simulation
vth = 20  # Soma spike threshold
dth = 5   # Dendrite spike threshold
dv = 0.95  # Soma decay 
dd = 0.95  # Dendrite decay
sd = 1  # Dendritic current scaling (1.0 in 10-bit fixed-point)
pt = 200      # Plateau duration (100 timesteps = 10 ms)
rt = 25       # Refractory period (20 timesteps = 2 ms)
rd = 100       # Dendritic refractory period (20 timesteps = 2 ms)
rdc = 0       # Dendritic refractory counter
bias = 0      # Neuron bias
h_plat = 0.55
v_reset = -8
# Input spikes (1 timestep each)
da0 = np.zeros(timesteps)  # Somatic input
da1 = np.zeros(timesteps)  # Dendritic input
da1[25:35] = 0.1  # Dendritic spike at 10 ms (timestep 100)
da0[100:110] = 1.5 # Somatic spike at 40 ms (timestep 400) - increased to 160 to ensure spike

a_adlif = 50 # in the adlif its range is [0.0, 1.0]
b_adlif = 1 # in the adlif its range is [0.0, 2.0]
rc_mcadlif = 0  # Refractory counter for mcadlif
alpha_adlif = np.exp(-dt / 5)
beta_adlif = np.exp(-dt / 10)

# Initialize neurons
lif = LIF(dv=dv, v_rest=0, v_reset=v_reset, v_thresh=vth, dt=dt, bias=bias)
adlif = adLIF(dv=dv, v_rest=0, v_reset=v_reset, v_thresh=vth, dt=dt, a=a_adlif, b=b_adlif, alpha_adlif=alpha_adlif, beta_adlif=beta_adlif)
mclif = MC(soma=LIF(dv=dv, v_rest=0, v_reset=v_reset, v_thresh=vth, dt=dt, bias=bias), 
           dendrites=[dendrite(num_neurons=1, h_plat=h_plat, dendrite_decay=dd, dth=dth, rt=pt, bias=bias)])
mcadlif = MC(soma=adLIF(dv=dv, v_rest=0, v_reset=v_reset, v_thresh=vth, dt=dt, a=a_adlif, b=b_adlif, alpha_adlif=alpha_adlif, beta_adlif=beta_adlif), 
             dendrites=[dendrite(num_neurons=1, h_plat=h_plat, dendrite_decay=dd, dth=dth, rt=pt, bias=bias)])

# Data storage for plotting
v_soma_history = np.zeros(timesteps)
v_lif_history = np.zeros(timesteps)
ud_dend_history = np.zeros(timesteps)
v_dend_history = np.zeros(timesteps)
v_dend_mcadlif_history = np.zeros(timesteps)
v_mcadlif_history = np.zeros(timesteps)
v_mclif_history = np.zeros(timesteps)
# Simulation loop
for t in range(timesteps):
    da_soma = da0[t]
    da_dend = da1[t]
    v_soma, v_dend, soma_spike, dend_spike = mclif.step(da_soma, da_dend)
    v_mclif_history[t] = v_soma
    v_dend_history[t] = v_dend
    v_lif, lif_spike = lif.step(da_soma)
    v_lif_history[t] = v_lif
    v_mcadlif, v_dend_mcadlif, soma_spike_mcadlif, dend_spike_mcadlif = mcadlif.step(da_soma, da_dend)
    v_mcadlif_history[t] = v_mcadlif
    v_dend_mcadlif_history[t] = v_dend_mcadlif
    u_adlif, adlif_spike = adlif.step(da_soma)
    v_soma_history[t] = v_soma
    ud_dend_history[t] = v_dend_mcadlif
    u_adlif_history = v_mcadlif_history
    w_adlif_history = np.zeros(timesteps)  # Placeholder, as w is not stored in this simple implementation
    vtot_mcadlif_history = v_mcadlif_history + v_dend_mcadlif_history

# Create time axis in milliseconds
time_ms = np.arange(0, timesteps) * dt

# Plot results
plt.figure(figsize=(12, 10))

plot_data = [
    (v_soma_history, 'g', 'Soma Potential (v)', 'Voltage', vth),
    (v_lif_history, 'orange', 'LIF Neuron Voltage (vm)', 'Voltage', vth),
    (ud_dend_history, 'c', 'Dendritic Current (ud)', 'Voltage', vth),
    (vtot_mcadlif_history, 'g', 'adLIF + Dendritic Current (u + ud)', 'Voltage', vth),
    (u_adlif_history, 'orange', 'adLIF Neuron Voltage (u)', 'Voltage', vth),
    (ud_dend_history, 'c', 'Dendritic Current (ud)', 'Voltage', vth),
    # (w_adlif_history, 'r', 'adLIF Adaptation Variable (w)', 'Voltage', None),
]

num_plots = 3 # 6
for i in range(num_plots):
    data, color, label, ylabel, threshold = plot_data[i]
    plt.subplot(num_plots, 1, i + 1)
    plt.plot(time_ms, data, color, label=label, linewidth=5)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        plt.ylim(-threshold/2, threshold*1.2)
    plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('neuron_simulation.png', dpi=300)
plt.show()