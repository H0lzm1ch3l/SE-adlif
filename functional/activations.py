
import torch

# Forward pass: Heaviside function
# Backward pass: Override Dirac Delta with gradient of fast sigmoid
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, k=25):
        ctx.save_for_backward(mem) # store the membrane potential for use in the backward pass
        ctx.k = k
        out = (mem > 0).float() # Heaviside on the forward pass: Eq(1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors  # retrieve membrane potential
        grad = grad_output / (ctx.k * torch.abs(mem) + 1.0) ** 2  # gradient of fast sigmoid on backward pass: Eq(4)
        return grad, None

# Forward pass: Heaviside function
# Backward pass: Override Dirac Delta with gradient of sigmoid    
class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, k=25):
        ctx.save_for_backward(mem)
        ctx.k = k
        out = (mem > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors
        grad = grad_output * ctx.k * torch.exp(-ctx.k * mem) / ((torch.exp(-ctx.k * mem) + 1) ** 2)
        return grad, None
    
# Forward pass: Heaviside function
# Backward pass: Override Dirac Delta with gradient of SLAYER function 
class SLAYER(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem, alpha=2.0, c=0.5):
        ctx.save_for_backward(mem)
        ctx.alpha = alpha
        ctx.c = c
        out = (mem > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (mem,) = ctx.saved_tensors
        grad = grad_output * (ctx.c * ctx.alpha) / (2 * torch.exp(ctx.alpha * torch.abs(mem)))
        return grad, None, None
