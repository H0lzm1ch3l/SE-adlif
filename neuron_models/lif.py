
class LIF:

    def __init__(self, vth=1.0, dv=0.95, v_reset=0.0, bias=0.0):
        self.vth = vth
        self.dv = dv
        self.v_reset = v_reset
        self.v = 0.0
        self.bias = bias
        self.rc = 0  # Refractory counter
    
    def step(self, input):
        spike = 0
        self.v = self.v * self.dv + input + self.bias
        if self.v >= self.vth:
            self.v = self.v_reset
            spike = 1
        return self.v, spike  # No spike
