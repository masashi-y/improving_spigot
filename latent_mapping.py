import torch
from sparsemax import Sparsemax

from spigot.algorithms.krucker import project_onto_knapsack_constraint_batch


def _argmax(x):
    _, feature_size = x.size()
    return torch.nn.functional.one_hot(
        torch.max(x, dim=-1).indices, num_classes=feature_size,
    ).float()


class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _argmax(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class StraightThroughSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _argmax(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        with torch.enable_grad():
            (grad_output,) = torch.autograd.grad(
                torch.softmax(x, dim=-1), x, grad_output
            )

        return grad_output, None


class SPIGOT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        results = _argmax(x)
        ctx.save_for_backward(results)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        (predictions,) = ctx.saved_tensors

        norm = torch.norm(grad_output, dim=-1)
        scale = torch.ones_like(norm)
        cond = norm > 1.0
        scale[cond] = 1.0 / norm[cond]

        target = -predictions + scale.unsqueeze(dim=-1) * grad_output

        projected = project_onto_knapsack_constraint_batch(target)
        # output = predictions - projected
        output = projected - predictions
        return output
        # output_scaled = (
        #     norm.unsqueeze(dim=-1)
        #     * output
        #     / (torch.norm(output, dim=-1, keepdim=True) + 1e-12)
        # )
        # return output_scaled
        # return torch.max(output, torch.zeros_like(output_scaled))


GAUSSIAN, UNIFORM = None, None


def init_noises(K, device=None):
    global GAUSSIAN, UNIFORM
    GAUSSIAN = torch.randn((K, K), device=device)
    UNIFORM = torch.rand((K, K), device=device)


class Noise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, noise):
        assert noise is not None
        ctx.save_for_backward(noise)
        return _argmax(x)

    @staticmethod
    def backward(ctx, grad_output):
        (noise,) = ctx.saved_tensors
        return torch.matmul(grad_output, noise), None


latent_mappings = {
    "spigot": SPIGOT.apply,
    "sparsemax": Sparsemax(dim=-1),
    "softmax": torch.nn.Softmax(dim=-1),
    "ste": StraightThrough.apply,
    "sts": StraightThroughSoftmax.apply,
    "gaussian": lambda x: Noise.apply(x, GAUSSIAN),
    "uniform": lambda x: Noise.apply(x, UNIFORM),
}
