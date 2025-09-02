import numpy as np
import matplotlib.pyplot as plt

# Saturate value to 24-bit signed range
def saturate24(x):
    if x > 0x7FFFFF:  # Maximum positive value (2^23 - 1)
        return 0x7FFFFF
    elif x < -0x800000:  # Minimum negative value (-2^23)
        return -0x800000
    return int(x)

# Constants (scaled to fixed-point representation)
dt = 0.1  # ms per timestep
timesteps = 800  # 60 ms simulation
vth = 20000  # Soma spike threshold
dth = 5000   # Dendrite spike threshold
dv = int(0.95 * (1 << 10))  # Soma decay (0.99 in 10-bit fixed-point)
dd = int(0.98 * (1 << 10))  # Dendrite decay
sd = 1 << 10  # Dendritic current scaling (1.0 in 10-bit fixed-point)
up = 10000    # Plateau potential
pt = 200      # Plateau duration (100 timesteps = 10 ms)
rt = 100       # Refractory period (20 timesteps = 2 ms)
rd = 100       # Dendritic refractory period (20 timesteps = 2 ms)
rdc = 0       # Dendritic refractory counter
bias = 0      # Neuron bias
h_plat = 210
# Input spikes (1 timestep each)
da0 = np.zeros(timesteps)  # Somatic input
da1 = np.zeros(timesteps)  # Dendritic input
da1[200:210] = 1  # Dendritic spike at 10 ms (timestep 100)
da0[400:401] = 160 # Somatic spike at 40 ms (timestep 400) - increased to 160 to ensure spike

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
alpha_adlif = np.exp(-dt / 10)
beta_adlif = np.exp(-dt / 100)

u_adlif_history = np.zeros(timesteps)
w_adlif_history = np.zeros(timesteps)

vtot_admclif = 0  # Total soma potential for adLIF
vtot_admclif_history = np.zeros(timesteps)

lif_spike_train = np.zeros(timesteps)
mclif_soma_spike_train = np.zeros(timesteps)
mclif_dend_spike_train = np.zeros(timesteps)
adlif_spike_train = np.zeros(timesteps)
admclif_spike_train = np.zeros(timesteps)

# Main simulation loop
for t in range(timesteps):
    # =========================================================================
    # Reference LIF Neuron (without dendritic components)
    # =========================================================================
    if rc_lif > 0:
        rc_lif -= 1
    else:
        # Apply decay and inputs
        v_lif = (v_lif * dv) >> 10
        v_lif = saturate24(v_lif + (int(da0[t]) << 6))
        v_lif = saturate24(v_lif + bias)
        
        # Check for spike
        if v_lif >= vth:
            v_lif = 0
            rc_lif = rt
            lif_spike_train[t] = 1
    
    v_lif_history[t] = v_lif

    # =========================================================================
    # Multi-compartment Neuron (based on microcode)
    # =========================================================================
    # Pass0: Soma update and refractory handling
    vtot_mclif = 0
    if rc_mclif > 0:
        rc_mclif -= 1
        # During refractory, only update refractory counter
        pass
    else:
        # Scale input, apply decay, and add bias
        scaled_da0 = int(da0[t]) << 6
        v_soma = (v_soma * dv) >> 10
        v_soma = saturate24(v_soma + scaled_da0)
        v_soma = saturate24(v_soma + bias)
    
    # Dendrite0: Update if dendrite is not active
    # Process dendritic input
    if rdc > 0:
        rdc -= 1
        scaled_da1 = 0  # No dendritic input during refractory
    else:
        scaled_da1 = int(da1[t]) << 6
    h_dend = (h_dend * dd) >> 10
    h_dend = saturate24(h_dend + scaled_da1)
    
    # Dendrite1: Update dendritic current
    ud_dend = (ud_dend * dd) >> 10
    # Add scaled h to ud (sd = 1.0 in fixed point)
    ud_dend = saturate24(ud_dend + ((h_dend * sd) >> 10))
    
    # Cases: Dendritic plateau logic
    print(f"Time {t}, Soma Voltage: {v_soma}, Dendrite Current: {ud_dend}, Active: {ac_dend}, Plateau Counter: {pc_dend}, Refractory Counter: {rc_mclif}")
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
    
    # Pass2: Apply dendritic current to soma and check for spike
    if rc_mclif == 0:  # Only if not in refractory

        vtot_mclif = saturate24(v_soma + ud_dend)

        if vtot_mclif >= vth:
            # Spike occurred
            v_soma = 0
            # ud = 0  # Reset dendritic current
            ac_dend = 0  # Deactivate dendrite
            pc_dend = 0  # Reset plateau counter
            rc_mclif = rt  # Enter refractory period
            mclif_soma_spike_train[t] = 1

    # =========================================================================
    # Adaptive LIF Neuron (based on Baronig et. al. 2024)
    # =========================================================================
    prev_spike = adlif_spike_train[t - 1] if t > 0 else 0
    u_adlif = alpha_adlif * u_adlif + (1 - alpha_adlif) * (-w_adlif + da0[t])
    w_adlif = beta_adlif * w_adlif + (1 - beta_adlif) * (a_adlif * u_adlif + b_adlif * prev_spike)

    # check spike
    if u_adlif >= 1.0:
        # u_adlif = 0
        adlif_spike_train[t] = 1

    # =========================================================================
    # Record state
    u_adlif_history[t] = u_adlif
    w_adlif_history[t] = w_adlif
    v_soma_history[t] = vtot_mclif
    ud_dend_history[t] = ud_dend
    vtot_mclif = 0

# Create time axis in milliseconds
time_ms = np.arange(0, timesteps) * dt

# Plot results
plt.figure(figsize=(12, 10))

# Plot 1: LIF Neuron Voltage
plt.subplot(5, 1, 1)
plt.plot(time_ms, v_lif_history, 'g', label='LIF Neuron Voltage (vm)', linewidth=5)
plt.ylabel('Voltage')
plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Threshold')
# plt.axvline(x=40, color='r', linestyle='--', alpha=0.3)
plt.ylim(0, 12000 * 1.1)  # Scale to 20% above threshold
plt.xticks([])  # Remove x-axis ticks and labels
plt.yticks([])  # Remove x-axis ticks and labels
# plt.grid(True)
plt.legend(loc='upper right', fontsize=14)

# Plot 2: Dendritic Potential
plt.subplot(5, 1, 2)
plt.plot(time_ms, ud_dend_history, 'b', label='Dendritic Current (ud)', linewidth=5)
plt.subplots_adjust(hspace=6)
plt.ylabel('Voltage')
plt.axhline(y=dth, color='r', linestyle='--', alpha=0.5, label='Dendritic Threshold')
# plt.axvline(x=20, color='r', linestyle='--', alpha=0.3)
# plt.axvline(x=40, color='r', linestyle='--', alpha=0.3)
plt.ylim(0, up * 1.4)  # Scale to 20% above threshold
plt.xticks([])  # Remove x-axis ticks and labels
plt.yticks([])  # Remove x-axis ticks and labels
# plt.grid(True)
plt.legend(loc='upper right', fontsize=14)

# Plot 3: Soma Potential (Multi-compartment)
plt.subplot(5, 1, 3)
plt.plot(time_ms, v_soma_history, 'r', label='Soma Potential (v)', linewidth=5)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage')
plt.axhline(y=vth, color='r', linestyle='--', alpha=0.5, label='Threshold')
# plt.axvline(x=40, color='r', linestyle='--', alpha=0.3)
plt.ylim(0, vth * 1.1)  # Scale to 20% above threshold
plt.xticks([])  # Remove x-axis ticks and labels
plt.yticks([])  # Remove x-axis ticks and labels
# plt.grid(True)
plt.legend(loc='upper right', fontsize=14)

# Plot 4: adLIF Neuron Voltage
plt.subplot(5, 1, 4)
plt.plot(time_ms, u_adlif_history, 'm', label='adLIF Neuron Voltage (u)', linewidth=5)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage')
plt.axhline(y=vth, color='r', linestyle='--', alpha=0.5, label='Threshold')
# plt.axvline(x=40, color='r', linestyle='--', alpha=0.3)
plt.ylim(-1, 2)  # Scale to 20% above threshold
plt.xticks([])  # Remove x-axis ticks and labels
plt.yticks([])  # Remove x-axis ticks and labels
# plt.grid(True)
plt.legend(loc='upper right', fontsize=14)

# Plot 5: adLIF Neuron Adaptation Variable
plt.subplot(5, 1, 5)
plt.plot(time_ms, w_adlif_history, 'c', label='adLIF Adaptation Variable (w)', linewidth=5)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage')
# plt.axhline(y=vth, color='r', linestyle='--', alpha=
# plt.axvline(x=40, color='r', linestyle='--', alpha=0.3)
plt.ylim(-1, 2)  # Scale to 20% above threshold
plt.xticks([])  # Remove x-axis ticks and labels
plt.yticks([])  # Remove x-axis ticks and labels
# plt.grid(True)
plt.legend(loc='upper right', fontsize=14)

plt.tight_layout()
plt.savefig('neuron_simulation.png', dpi=300)
plt.show()