import numpy as np

class adLIF:
    
    def __init__(self, dv=0.95, v_rest=0, v_reset=0, v_thresh=1.0, dt=1.0, a=0.0, b=0.0, alpha_adlif=0.85, beta_adlif=0.90):
        self.dv = dv
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.a = a
        self.b = b
        self.alpha_adlif = alpha_adlif
        self.beta_adlif = beta_adlif
        self.u = 0.0  # Membrane potential
        self.w = 0.0  # Adaptation variable
        self.rc = 0   # Refractory counter
        self.spike = 0
        
    def step(self, input):
        prev_spike = self.spike
        self.u = self.alpha_adlif * self.u + (1 - self.alpha_adlif) * (-self.w + (input if self.rc==0 else 0))
        self.w = self.beta_adlif * self.w + (1 - self.beta_adlif) * (self.a * self.u + self.b * prev_spike)
        
        # Check for spike
        if self.u >= self.v_thresh:
            self.spike = 1
            # self.u = self.v_reset  # Optionally reset membrane potential on spike
        else:
            self.spike = 0
        
        return self.u, self.spike
    