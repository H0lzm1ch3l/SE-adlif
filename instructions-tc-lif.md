Here are the comprehensive implementation instructions for building the Two-Compartment Leaky Integrate-and-Fire (TC-LIF) network to replicate the Spiking Google Speech Commands (SSC) results. 

These instructions extract the exact mathematical formulations, parameter initializations, and network topologies directly from the study so you can construct the model architecture.

### **1. Core TC-LIF Neuron Dynamics**
[cite_start]The TC-LIF neuron separates the membrane state into two interacting compartments: a somatic compartment ($\mathcal{U}^S$) for short-term memory and a dendritic compartment ($\mathcal{U}^D$) for long-term memory[cite: 162, 163].

**Parameters to Initialize per Layer:**
* [cite_start]**$c_1$ and $c_2$**: Learnable parameters used to control the coupling effects between the two compartments[cite: 158].
* [cite_start]**$\beta_1$ and $\beta_2$**: The interaction coefficients derived from $c_1$ and $c_2$[cite: 157]. [cite_start]You must initialize these values in the "second quadrant" to ensure stable training and avoid gradient vanishing or exploding[cite: 291]. Apply a sigmoid function $\sigma(\cdot)$ during the forward pass to restrict the coefficients:
    [cite_start]$$\beta_1 \equiv -\sigma(c_1)$$ [cite: 157]
    [cite_start]$$\beta_2 \equiv \sigma(c_2)$$ [cite: 157]
    [cite_start]*(This ensures $\beta_1 \in (-1, 0)$ and $\beta_2 \in (0, 1)$ [cite: 158, 291])*
* [cite_start]**$\gamma$**: A scaling factor that governs the partial reset of the dendritic compartment triggered by a backpropagating spike from the soma[cite: 161, 163].
* [cite_start]**$\mathcal{V}_{th}$**: The neuronal firing threshold[cite: 110, 154].

**Timestep Updates ($t$):**
For each timestep, calculate the total input current $\mathcal{I}[t]$ from incoming spikes:
[cite_start]$$\mathcal{I}[t] = \sum_i \omega_i x_i[t] + b$$ [cite: 105, 109]

Update the dendritic membrane potential (long-term memory):
[cite_start]$$\mathcal{U}^D[t] = \mathcal{U}^D[t-1] + \beta_1 \mathcal{U}^S[t-1] + \mathcal{I}[t] - \gamma \mathcal{S}[t-1]$$ [cite: 152]

Update the somatic membrane potential (short-term memory):
[cite_start]$$\mathcal{U}^S[t] = \mathcal{U}^S[t-1] + \beta_2 \mathcal{U}^D[t] - \mathcal{V}_{th} \mathcal{S}[t-1]$$ [cite: 153]

Generate the output spike using the Heaviside step function $\Theta(\cdot)$:
[cite_start]$$\mathcal{S}[t] = \Theta(\mathcal{U}^S[t] - \mathcal{V}_{th})$$ [cite: 154]

[cite_start]*(Note: The membrane decaying factors $\alpha_1$ and $\alpha_2$ are explicitly dropped (set to 1) in the TC-LIF model to circumvent the rapid decay of memory[cite: 147, 148, 203].)*

---

### **2. Network Architecture (SSC Configurations)**
[cite_start]For the SSC task, the model is evaluated using two primary network topologies, both constrained to approximately 110.8K parameters[cite: 299].

* **Feed-forward SNN:** A purely feedforward architecture where temporal dynamics are handled entirely by the internal states of the TC-LIF neurons. [cite_start]This configuration achieved 63.46% accuracy[cite: 299].
* **Recurrent SNN (SRNN):** A recurrent architecture where lateral or feedback connections are added between neurons within the same layer or across layers. [cite_start]This configuration achieved 61.09% accuracy[cite: 299].

---

### **3. Backpropagation and Surrogate Gradient**
[cite_start]Because the step function $\Theta(\cdot)$ used for spike generation is non-differentiable[cite: 154], standard backpropagation fails. 

* [cite_start]**Algorithm:** You must implement the backpropagation-through-time (BPTT) algorithm coupled with surrogate gradients to perform credit assignment[cite: 113, 199].
* [cite_start]**Gradient Flow:** The TC-LIF model is specifically designed so that the partial derivative $\frac{\partial \mathcal{U}[j]}{\partial \mathcal{U}[j-1]}$ exceeds 1 for parameters in the second quadrant, inherently alleviating the vanishing gradient problem over extended temporal durations[cite: 184, 187].
* **Loss Function:** The network is optimized using a standard sample-averaged loss function:
    [cite_start]$$\mathcal{L}(\hat{\mathcal{S}}, \mathcal{S}) = \frac{1}{N} \sum_{n=1}^N \mathcal{L}(\hat{\mathcal{S}}_n, \mathcal{S}_n)$$ [cite: 117]

