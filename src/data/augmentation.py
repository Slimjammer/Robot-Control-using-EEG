import torch

def augment(x: torch.Tensor) -> torch.Tensor:
    """Stochastic augmentation: Gaussian noise + random time shift. Input: (B, T, C)"""
    x = x + 0.01 * torch.randn_like(x)
    shift = torch.randint(-20, 20, (1,)).item()
    return torch.roll(x, shifts=shift, dims=1)
