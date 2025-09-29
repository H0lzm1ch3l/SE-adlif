
from neuron_models.dendrite import dendrite
from neuron_models.lif import LIF

# class compartmentLIF(LIF):
#     def step(self, input):
#         self.v = self.v * self.dv + input + self.bias
#         return self.v, 0  # No spike
        
class MCLIF:
    
    def __init__(self, h_plat=0.55, dendrite_decay=0.95, dth=1.0, rt=5, bias=0.0,
                 vth=1.0, dv=0.95, v_reset=0.0):
        self.lif = LIF(vth=vth, dv=dv, v_reset=v_reset, bias=bias)
        self.dendrite = dendrite(h_plat=h_plat, dendrite_decay=dendrite_decay, dth=dth, rt=rt, bias=bias)
    
    def step(self, da_soma, da_dend):
        v_dend, dend_spike = self.dendrite.step(da_dend)
        v_soma, soma_spike = self.lif.step(da_soma + v_dend)
        return v_soma, v_dend, soma_spike, dend_spike