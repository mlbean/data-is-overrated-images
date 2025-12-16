import torch 


def quantise(zi, scale_factor=10, centers=(-1, 1)):
    """
    Soft quantisation of the latent representation zi to the given centers.
    Works for arbitrary size of zi and number of centers.
    
    NOTE on the scale factor from the original paper: Increasing s draws input
    values closer to the nearest center, but too large s can lead to numerical instability.   
    
    Example:
    >>> zi = torch.randn(4, 3, 2, 2)  # Example latent representation
    >>> centers = (-1, 0, 1)  # Example centers
    >>> zi_hat = quantise(zi, scale_factor=10, centers=centers)
    >>> print(zi_hat.shape)  # Should be (4, 3, 2, 2)
    """
    
    s = scale_factor
    c = torch.tensor(centers, device=zi.device, dtype=zi.dtype)  # shape (num_centers,)
    c = c.view(*([1]*(zi.ndim-1) + [-1]))  # shape (1, ..., 1, num_centers) to ensure correct broadcasting
    zi = zi.unsqueeze(-1)  # shape (B, C, H, W, num_centers)
    
    fraction = torch.softmax(-s * (zi - c)**2, dim=-1)  # shape (B, C, H, W, num_centers)
    zi_hat = torch.sum(c * fraction, dim=-1)
    return zi_hat
