import torch
import matplotlib.pyplot as plt
import numpy as np

# For now this the sech autograd function with the true gradient, not the surrogate gradient -> I will need to look into possible surrogate functions 
@staticmethod
class sech(torch.autograd.Function):
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return - torch.sinh(x) / (torch.cosh(x)**2)
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sech_x = - torch.sinh(x) / (torch.cosh(x)**2)
        grad_input = grad_output * (sech_x**2 - sech_x * torch.tanh(x))
        return grad_input
    
# def sech_fn(x):
#     return - torch.sinh(x) / (torch.cosh(x)**2)

def sech_fn(x):
    return 1 / torch.cosh(x)

# sech experiments:
amplitude = 1.0
omega = 3
x = torch.linspace(-10, 10, steps=100)
x.requires_grad = True
y = amplitude * sech_fn(x / omega)
# y = sech_fn(x)
dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)

data_x = x.detach().numpy()
data_y = y.detach().numpy()
data_dy_dx = dy_dx[0].detach().numpy()

# Plot the function along with its gradient
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(data_x, data_y, label="sech(x)", color='blue')
plt.title('sech Activation Function')
plt.xlabel('x')
plt.ylabel('sech(x)')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(data_x, data_dy_dx, label="d(sech)/dx", color='orange')
plt.title('Gradient of sech Function')
plt.xlabel('x')
plt.ylabel('d(sech)/dx')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()  
