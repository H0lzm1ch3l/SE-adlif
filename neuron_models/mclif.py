
# class compartmentLIF(LIF):
#     def step(self, input):
#         self.v = self.v * self.dv + input + self.bias
#         return self.v, 0  # No spike
        
class MC:
    
    def __init__(self, soma, dendrite):
        self.soma = soma
        self.dendrite = dendrite
    
    def step(self, da_soma, da_dend):
        v_dend, dend_spike = self.dendrite.step(da_dend)
        v_soma, soma_spike = self.lif.step(da_soma + v_dend)
        return v_soma, v_dend, soma_spike, dend_spike