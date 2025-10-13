import numpy as np
import matplotlib.pyplot as plt
import torch

def cubic_plateau(V, I, dt, V_peak=20, a=0.01):
    """
    Direct cubic function for plateau potential
    V: current membrane potential
    I: input current
    dt: timestep
    Returns: new membrane potential
    """
    # Cubic dynamics: dV/dt = I - a*(V-V_rest)*(V-V_th)*(V-V_peak)
    dV = I - a * V**2 * (V - V_peak)
    return V + dt * dV

def sigmoidal_plateau(V, I, dt, V_rest=-70, g_leak=0.1, g_Ca=0.5, 
                     E_Ca=120, V_half=-40, k=5):
    """
    Direct sigmoidal function modeling calcium channels
    """
    # Leak current
    leak_current = g_leak * (V - V_rest)
    
    # Calcium current with sigmoidal activation
    m_inf = 1.0 / (1.0 + np.exp(-(V - V_half) / k))
    calcium_current = g_Ca * m_inf * (V - E_Ca)
    
    # Total change
    dV = I - leak_current - calcium_current
    return V + dt * dV

def double_sigmoid_plateau(V, I, dt, V_plateau=20, 
                          threshold=-55, sharpness=0.5, persistence=0.95):
    """
    Uses two sigmoids: one for activation, one for persistence
    """
    # Activation sigmoid - determines if we cross threshold
    activation = 1.0 / (1.0 + np.exp(-(V - threshold) / sharpness))
    
    # Persistence sigmoid - keeps us in plateau once we're there
    if V > threshold:
        persistence_factor = persistence
    else:
        persistence_factor = 0.0
    
    # Combined effect
    target_V = V_plateau * max(activation, persistence_factor)
    
    # Smooth transition toward target
    tau = 5.0  # ms time constant
    dV = (target_V - V) / tau + I
    return V + dt * dV

def exponential_plateau(V, I, dt, V_rest=-70, V_plateau=20, 
                       threshold=-55, tau_rise=2.0, tau_decay=20.0):
    """
    Exponential transitions between states with explicit threshold
    """
    # Determine current state
    if V < threshold:
        # Moving toward rest
        target = V_rest
        tau = tau_decay
    else:
        # In plateau state
        target = V_plateau  
        tau = tau_rise
    
    # Exponential approach to target
    dV = (target - V) / tau + I
    new_V = V + dt * dV
    
    # Hysteresis: once above threshold, stay in plateau until forced down
    if V > threshold and new_V < threshold:
        # Resist falling below threshold without strong negative current
        if I >= 0:  # No strong negative current
            new_V = threshold + 0.1  # Keep just above threshold
    
    return new_V

def rate_based_plateau(V, I, dt, r=0.0, V_rest=-70, V_plateau=20,
                      threshold=-55, tau_V=5.0, tau_r=50.0, alpha=0.1):
    """
    Includes a rate variable for additional dynamics
    V: membrane potential
    r: recovery variable (like calcium-activated potassium channels)
    """
    # Voltage dynamics
    if V > threshold:
        target_V = V_plateau
    else:
        target_V = V_rest
    
    dV = (target_V - V - alpha * r) / tau_V + I
    
    # Recovery dynamics (slower)
    dr = -r / tau_r  # Slow decay
    if V > threshold:
        dr += 0.01  # Accumulate during plateau
    
    new_V = V + dt * dV
    new_r = r + dt * dr
    
    return new_V, new_r

def simulate_plateau(stimulus, plateau_fn, V0=0, dt=0.1, **kwargs):
    """
    Simulate neuron dynamics given a stimulus and plateau function
    stimulus: input current over time
    plateau_fn: function defining the plateau dynamics
    V0: initial membrane potential
    dt: timestep
    kwargs: additional parameters for the plateau function
    Returns: membrane potential over time
    """
    V = torch.zeros_like(stimulus)
    V[0] = V0
    r = torch.zeros_like(stimulus)  # For rate-based model
    
    for i in range(1, len(stimulus)):
        if plateau_fn == rate_based_plateau:
            V[i], r[i] = plateau_fn(V[i-1], r[i-1], dt, I=stimulus[i], **kwargs)
        else:
            V[i] = plateau_fn(V[i-1], stimulus[i], dt, **kwargs)
    
    return V



# Example usage
dt = 0.1  # ms
t = torch.arange(0, 100, dt)  # 1000 ms total
stimulus = torch.zeros_like(t)
stimulus[200:300] = 5  # Input current pulse from 20 ms to 30 ms
stimulus.requires_grad = True

V = simulate_plateau(stimulus, sigmoidal_plateau, dt=dt)