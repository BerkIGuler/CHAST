import torch


def complex_grid_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex (..., K, L) tensor into a real (..., 2, K, L) tensor.
    Channel 0 is real part, channel 1 is imaginary part.
    """
    if not torch.is_complex(x):
        raise TypeError(f"Expected a complex tensor, got dtype={x.dtype}.")
    if x.ndim < 2:
        raise ValueError(f"Expected shape (..., K, L), got {tuple(x.shape)}.")
    # Use float32 for downstream CNN/Transformer.
    return torch.stack((x.real, x.imag), dim=-3).to(dtype=torch.float32)


def nmse_db(pred_2ch: torch.Tensor, target_2ch: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute NMSE in dB for 2-channel real/imag tensors shaped (..., 2, K, L).
    Returns a scalar tensor (mean over all leading dims).
    """
    if pred_2ch.shape != target_2ch.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred_2ch.shape)} target={tuple(target_2ch.shape)}")
    if pred_2ch.size(-3) != 2:
        raise ValueError(f"Expected channel dim size 2 at -3, got {pred_2ch.size(-3)}")

    err = pred_2ch - target_2ch
    dims = tuple(range(1, pred_2ch.ndim))
    num = (err * err).sum(dim=dims)
    den = (target_2ch * target_2ch).sum(dim=dims).clamp_min(eps)
    nmse = (num / den).clamp_min(eps)
    return 10.0 * torch.log10(nmse).mean()

