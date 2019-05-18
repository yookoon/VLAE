import torch


def clip_grad(gradient, clip_value):
    """ clip between clip_min and clip_max
    """
    return torch.clamp(gradient, min=-clip_value, max=clip_value)

def clip_grad_norm(gradient, clip_value):
    norm = (gradient**2).sum(-1)
    divisor = torch.max(torch.ones_like(norm).cuda(), norm / clip_value)
    return gradient / divisor.unsqueeze(-1)
