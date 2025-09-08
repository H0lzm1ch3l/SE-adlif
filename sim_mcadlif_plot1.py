import numpy as np
import matplotlib.pyplot as plt

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
h_plat = 0.5
# Input spikes (1 timestep each)
da0 = np.zeros(timesteps)  # Somatic input
da1 = np.zeros(timesteps)  # Dendritic input
da1[25:35] = 0.1  # Dendritic spike at 10 ms (timestep 100)
da0[100:110] = 1.5 # Somatic spike at 40 ms (timestep 400) - increased to 160 to ensure spike

# Initialize state variables for multi-compartment neuron
v_soma = 0        # Soma membrane potential
vtot_mclif = 0  # Total soma potential 
h_dend = 0        # Dendrite state variable
ud_dend = 0       # Dendrite current
ac_dend = 0       # Active dendrite flag
pc_dend = 0       # Plateau counter
rc_mclif = 0       # Refractory counter
v_soma_history = np.zeros(timesteps)      # Soma potential history
ud_dend_history = np.zeros(timesteps)     # Dendrite current history

# Initialize state variables for LIF neuron (point neuron)
v_lif = 0
rc_lif = 0
v_lif_history = np.zeros(timesteps)

# Initialize state variables for adLIF neuron (oscillating potential)
u_adlif = 0
w_adlif = 0
a_adlif = 50 # in the adlif its range is [0.0, 1.0]
b_adlif = 1 # in the adlif its range is [0.0, 2.0]
alpha_adlif = np.exp(-dt / 5)
beta_adlif = np.exp(-dt / 10)
rc_mcadlif = 0  # Refractory counter for mcadlif

u_adlif_history = np.zeros(timesteps)
w_adlif_history = np.zeros(timesteps)

vtot_mcadlif = 0  # Total soma potential for adLIF
vtot_mcadlif_history = np.zeros(timesteps)

lif_spike_train = np.zeros(timesteps)
mclif_soma_spike_train = np.zeros(timesteps)
mclif_dend_spike_train = np.zeros(timesteps)
adlif_spike_train = np.zeros(timesteps)
mcadlif_spike_train = np.zeros(timesteps)

# Main simulation loop
for t in range(timesteps):
    # =========================================================================
    # Reference LIF Neuron (without dendritic components)
    # =========================================================================
    if rc_lif > 0:
        rc_lif -= 1
    else:
        # Apply decay and inputs
        v_lif = v_lif * dv + da0[t] + bias
        # Check for spike
        if v_lif >= vth:
            rc_lif = rt
            lif_spike_train[t] = 1
    
    v_lif_history[t] = v_lif

    # =========================================================================
    # Adaptive LIF Neuron (based on Baronig et. al. 2024)
    # =========================================================================
    prev_spike = adlif_spike_train[t - 1] if t > 0 else 0
    u_adlif = alpha_adlif * u_adlif + (1 - alpha_adlif) * (-w_adlif + (da0[t]*50 if rc_mcadlif==0 else 0))
    w_adlif = beta_adlif * w_adlif + (1 - beta_adlif) * (a_adlif * u_adlif + b_adlif * prev_spike)
    # print(f"Time {t}, u: {u_adlif}, w: {w_adlif}, input: {da0[t]}, prev_spike: {prev_spike}")

    # check spike
    if u_adlif >= 1.0:
        # u_adlif = 0
        adlif_spike_train[t] = 1

    # =========================================================================
    # Multi-compartment Neuron (based on microcode)
    # =========================================================================
    # Pass0: Soma update and refractory handling
    vtot_mclif = 0
    if rc_mclif > 0:
        rc_mclif -= 1
        # During refractory, only update refractory counter
        v_soma = v_soma * dv
        pass
    else:
        # Scale input, apply decay, and add bias
        v_soma = v_soma * dv + da0[t] + bias
    
    # Dendrite0: Update if dendrite is not active
    # Process dendritic input
    if rdc > 0:
        rdc -= 1
        dend_da = 0  # No dendritic input during refractory
    else:
        dend_da = da1[t]
    h_dend = h_dend * dd + dend_da
    
    # Dendrite1: Update dendritic current
    # Add scaled h to ud (sd = 1.0 in fixed point)
    ud_dend = ud_dend * dd + (h_dend * sd)
    
    # Cases: Dendritic plateau logic
    # print(f"Time {t}, Soma Voltage: {v_soma}, Dendrite Current: {ud_dend}, Active: {ac_dend}, Plateau Counter: {pc_dend}, Refractory Counter: {rc_mclif}")
    if ud_dend >= dth and ac_dend == 0 and pc_dend == 0 and rdc == 0: # dendrite activated
        # Initiate plateau
        # ud = up
        pc_dend = pt
        ac_dend = 1
        h_dend = h_plat
        mclif_dend_spike_train[t] = 1
    
    # Update plateau counter
    if ac_dend == 1:
        # ud = up
        h_dend = h_plat
        pc_dend -= 1
        if pc_dend <= 0:
            # End plateau
            # ud = 0
            h_dend = 0
            ac_dend = 0
            rdc = rd  # Enter dendritic refractory period
    

    vtot_mcadlif = u_adlif + ud_dend
    if vtot_mcadlif >= vth:
        mcadlif_spike_train[t] = 1
        rc_mcadlif = rt

    vtot_mclif = v_soma + ud_dend
    # Pass2: Apply dendritic current to soma and check for spike
    if rc_mclif == 0:  # Only if not in refractory
        # TODO: Add a separate dendritic compartment for the adLIF neuron
        if vtot_mclif >= vth:
            # Spike occurred
            # v_soma = 0
            # ud = 0  # Reset dendritic current
            # ac_dend = 0  # Deactivate dendrite
            # pc_dend = 0  # Reset plateau counter
            rc_mclif = rt  # Enter refractory period
            mclif_soma_spike_train[t] = 1


    # =========================================================================
    # Record state
    u_adlif_history[t] = u_adlif
    w_adlif_history[t] = w_adlif
    vtot_mcadlif_history[t] = vtot_mcadlif
    v_soma_history[t] = vtot_mclif
    ud_dend_history[t] = ud_dend
    # vtot_mclif = 0

# Create time axis in milliseconds
time_ms = np.arange(0, timesteps) * dt

# Plot results
plt.figure(figsize=(12, 20))

plot_data = [
    (v_soma_history, 'g', 'Soma Potential (v)', 'Voltage', vth),
    (v_lif_history, 'orange', 'LIF Neuron Voltage (vm)', 'Voltage', vth),
    (ud_dend_history, 'c', 'Dendritic Current (ud)', 'Voltage', vth),
    (vtot_mcadlif_history, 'g', 'adLIF + Dendritic Current (u + ud)', 'Voltage', vth),
    (u_adlif_history, 'orange', 'adLIF Neuron Voltage (u)', 'Voltage', vth),
    (ud_dend_history, 'c', 'Dendritic Current (ud)', 'Voltage', vth),
    # (w_adlif_history, 'r', 'adLIF Adaptation Variable (w)', 'Voltage', None),
]

for i, (data, color, label, ylabel, threshold) in enumerate(plot_data, 1):
    plt.subplot(6, 1, i)
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