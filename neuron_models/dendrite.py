import numpy as np

# TODO: Implement sech(...) activation function dendrite
class sech:
    def __init__(self):
        pass
    def __call__(self, x):
        return - np.sinh(x) / (np.cosh(x)**2)

# TODO: This implementation is not super practical for DL because of the many control flow usages
class dendrite:
    def __init__(self, num_neurons, h_plat=0.55, dendrite_decay=0.95, dth=1.0, rt=5, bias=0.0):
        self.num_neurons = num_neurons
        self.h_plat = h_plat
        self.dendrite_decay = dendrite_decay
        self.dth = dth 
        self.rt = rt
        self.bias = bias
        self.h = np.zeros(num_neurons)  # Dendrite state variable
        self.ud = np.zeros(num_neurons)  # Dendrite current
        self.ac = np.zeros(num_neurons)  # Active dendrite flag
        self.pc = np.zeros(num_neurons)  # Plateau counter
        self.rc = np.zeros(num_neurons)  # Refractory counter
        self.spike = np.zeros(num_neurons)  # Spike flag
        
    def step(self, input):
        # Update refractory counter
        self.rc = np.maximum(0, self.rc - 1)
        
        # Update dendrite current
        self.ud = self.dendrite_decay * self.ud + (1 - self.dendrite_decay) * (input * (self.rc == 0))
        
        # Update dendrite state variable
        self.h = self.h + self.ud
        
        # Check for spike
        prev_spike = self.spike.copy()
        self.spike = (self.h >= self.dth).astype(float)
        
        # Handle spike event
        for i in range(self.num_neurons):
            if self.spike[i] == 1 and prev_spike[i] == 0:
                self.h[i] = self.h_plat
                self.ac[i] = 1
                self.pc[i] = self.rt
                self.rc[i] = self.rt
        
        # Update plateau counter and active flag
        for i in range(self.num_neurons):
            if self.ac[i] == 1:
                if self.pc[i] > 0:
                    self.pc[i] -= 1
                else:
                    self.ac[i] = 0
        
        # Reset h if not active
        for i in range(self.num_neurons):
            if self.ac[i] == 0:
                self.h[i] = max(0, self.h[i])
        
        return self.h, self.spike