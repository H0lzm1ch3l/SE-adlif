
# TODO: Implement sech(...) activation function dendrite
class sech:
    def __init__(self):
        pass
    def __call__(self, x):
        pass

# TODO: This implementation is not super practical for DL because of the many control flow usages
class dendrite:
    def __init__(self, h_plat=0.55, dendrite_decay=0.95, dth=1.0, rt=5, bias=0.0):
        self.h_plat = h_plat
        self.bias = bias
        self.dendrite_decay = dendrite_decay
        self.dth = dth
        self.rt = rt
        self.v_dend = 0.0
        self.h = 0.0
        self.rc = 0  # Refractory counter
        self.ac = 0  # Active counter

    def step(self, input):
        if self.rc > 0:
            self.rc -= 1
            dend_da = 0  # No dendritic input during refractory
        else:
            dend_da = input
        self.h = self.h * self.dendrite_decay + dend_da

        # Update dendritic current
        self.v_dend = self.v_dend * self.dendrite_decay + (self.h * 1.0)  # sd = 1.0 in fixed point

        # Cases: Dendritic plateau logic
        if self.v_dend >= self.dth and self.ac == 0 and self.rc == 0: # dendrite activated
            # Initiate plateau
            self.ac = 1
            self.h = self.h_plat
            spike = 1
        else:
            spike = 0

        # Update plateau counter
        if self.ac == 1:
            self.h = self.h_plat
            self.ac -= 1
            if self.ac <= 0:
                # End plateau
                self.h = 0
                self.ac = 0
                self.rc = self.rt  # Enter dendritic refractory period

        return self.v_dend, spike
