
import torch

# Forward pass: Heaviside function
# Backward pass: Override Dirac Delta with gradient of fast sigmoid
@staticmethod
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
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.k * torch.abs(mem) + 1.0) ** 2  # gradient of fast sigmoid on backward pass: Eq(4)
        return grad, None
    
@staticmethod
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
        grad_input = grad_output.clone()
        grad = grad_input * (ctx.c * ctx.alpha) / (2 * torch.exp(ctx.alpha * torch.abs(mem)))
        return grad, None, None
