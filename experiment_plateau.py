import numpy as np
import matplotlib.pyplot as plt
import torch

def cubic_plateau(V, I, dt, V_rest=-70, V_th=-55, V_peak=20, a=0.01):
    """
    Direct cubic function for plateau potential
    V: current membrane potential
    I: input current
    dt: timestep
    Returns: new membrane potential
    """
    # Cubic dynamics: dV/dt = I - a*(V-V_rest)*(V-V_th)*(V-V_peak)
    dV = I - a * (V - V_rest) * (V - V_th) * (V - V_peak)
    return V + dt * dV

def simulate_cubic_plateau(stimulus, V0=-70, dt=0.1, **params):
    """Simulate entire trace"""
    V = torch.zeros_like(stimulus)
    V[0] = V0
    
    for t in range(1, len(stimulus)):
        V[t] = cubic_plateau(V[t-1], stimulus[t], dt, **params)
    
    return V

# Example usage
dt = 0.1  # ms
t = torch.arange(0, 100, dt)  # 1000 ms total
stimulus = torch.zeros_like(t)
stimulus[200:300] = 5  # Input current pulse from 20 ms to 30 ms
stimulus.requires_grad = True

V = simulate_cubic_plateau(stimulus, V0=0, dt=dt, V_rest=0, V_th=15, V_peak=20, a=0.01)
# gradient
dy_dx = torch.autograd.grad(V, stimulus, torch.ones_like(V), create_graph=True)
data_t = t.detach().numpy()
data_V = V.detach().numpy()
data_dy_dx = dy_dx[0].detach().numpy()

# Plot the membrane potential and its gradient
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(data_t, data_V, label="Membrane Potential V(t)", color='blue')
plt.title('Cubic Plateau Neuron Model')
plt.xlabel('Time (ms)')
plt.ylabel('V (mV)')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(data_t, data_dy_dx, label="dV/dI", color='orange')
plt.title('Gradient of Membrane Potential w.r.t Input Current')
plt.xlabel('Time (ms)')
plt.ylabel('dV/dI')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
