import numpy as np
import matplotlib.pyplot as plt


def lif_neuron(x, v, threshold=1.0, dt=1.0, tau=10.0):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Args:
        v (float): Membrane potential of the neuron.
        threshold (float): Firing threshold.
        dt (float): Time step.
        tau (float): Membrane time constant.

    Returns:
        float: Updated membrane potential.
        bool: Whether the neuron fired.
    """
    # Update the membrane potential
    dv = (-(v / tau) + 1.0) * dt + x
    v += dv

    # Check for firing
    fired = v >= threshold
    if fired:
        v = 0.0  # Reset potential after firing

    return v, fired

def mclif_neuron(x_d, x_s, u, z, d, t, decay_u, decay_d, decay_t, threshold=1.0):
    # Update potentials 
    d = d * decay_d + x_d

    u = u * decay_u + x_s

    # check refractory
    if z:
        u = 0.0

    # Check for firing
    fired = u >= threshold
    if fired:
        u = 0.0  # Reset potential after firing

    return u, d, t, fired