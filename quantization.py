"""Post-training weight quantization for evaluating trained checkpoints under
different weight bit-widths.

Scheme (see cluster_project/README_quantization.md for the full write-up):
  - Only nn.Conv2d.weight and nn.Linear.weight tensors are quantized. Biases,
    BatchNorm affine parameters, and BatchNorm running statistics are left at
    full fp32 precision (standard PTQ practice: they are a tiny fraction of
    total parameters but disproportionately sensitive to quantization noise).
  - Symmetric, signed, uniform ("linear") quantization, per output channel by
    default (one scale per Conv2d out_channel / Linear output row), since
    weight magnitude ranges vary substantially across channels/layers.
  - "Fake" quantization: weights are snapped onto the b-bit grid but stored
    back as float32, so evaluation is a normal PyTorch forward pass. This
    isolates the accuracy effect of b-bit weight quantization; it does not
    by itself realize the memory/latency savings of real integer inference
    (which would additionally require integer GEMM kernels).
  - No calibration data and no activation quantization: only weights are
    quantized, per the task at hand, so no calibration statistics are
    needed (calibration is only required for activation quantization).
"""

from copy import deepcopy
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


QUANTIZABLE_TYPES = (nn.Conv2d, nn.Linear)


def quantize_tensor(
    w: torch.Tensor,
    bits: int,
    per_channel: bool = True,
    channel_dim: int = 0,
) -> torch.Tensor:
    """Symmetric signed uniform ("fake") quantization of a weight tensor.

    Args:
        w: Weight tensor to quantize.
        bits: Bit-width b. Uses qmax = 2**(b-1) - 1 signed levels
            (e.g. b=8 -> qmax=127, 255 total levels including zero).
        per_channel: Compute one scale per slice along `channel_dim` instead
            of a single scale for the whole tensor. Defaults to True.
        channel_dim: Dimension to treat as the output-channel axis for
            per-channel scaling (0 for both nn.Conv2d.weight and
            nn.Linear.weight). Defaults to 0.

    Returns:
        Quantized-then-dequantized weight tensor (same shape/dtype as `w`).
    """
    if bits >= 32:
        return w.clone()
    if bits < 2:
        raise ValueError(f"bits must be >= 2 for a usable signed range, got {bits}")

    qmax = 2 ** (bits - 1) - 1

    if per_channel:
        reduce_dims = [d for d in range(w.dim()) if d != channel_dim]
        amax = w.detach().abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = w.detach().abs().max()

    amax = amax.clamp_min(1e-12)
    scale = amax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


@torch.no_grad()
def quantize_model_(
    model: nn.Module,
    bits: int,
    per_channel: bool = True,
    keep_first_last_fp32: bool = False,
    layer_types: Tuple[type, ...] = QUANTIZABLE_TYPES,
) -> nn.Module:
    """Quantize a model's Conv2d/Linear weights in place.

    Args:
        model: Model to quantize (modified in place; also returned).
        bits: Bit-width to quantize to. `bits >= 32` is a no-op (fp32
            passthrough), useful as a baseline row in a sweep.
        per_channel: Use per-output-channel scales instead of one scale per
            tensor. Defaults to True.
        keep_first_last_fp32: If True, skip quantization for the first and
            last quantizable layer (common refinement: first/last layers are
            disproportionately accuracy-sensitive in aggressive low-bit
            quantization). Defaults to False (uniform bit-width everywhere).
        layer_types: Module types to quantize. Defaults to
            (nn.Conv2d, nn.Linear).

    Returns:
        The same model instance, with weights quantized in place.
    """
    targets: List[nn.Module] = [
        m for m in model.modules() if isinstance(m, layer_types)
    ]
    skip_ids = set()
    if keep_first_last_fp32 and targets:
        skip_ids = {id(targets[0]), id(targets[-1])}

    for module in targets:
        if id(module) in skip_ids:
            continue
        module.weight.data = quantize_tensor(
            module.weight.data, bits=bits, per_channel=per_channel, channel_dim=0
        )
    return model


def quantization_error(state_dict: Dict[str, torch.Tensor], bits: int) -> float:
    """Mean relative L2 error introduced by quantizing every Conv2d/Linear
    weight tensor in a raw state_dict, without needing a live model instance.
    Utility for sanity-checking / reporting quantization severity per bit-width.

    Args:
        state_dict: Raw state_dict (e.g. loaded from a `model.pt` checkpoint).
        bits: Bit-width to test.

    Returns:
        Mean of ||w_q - w||_2 / ||w||_2 over all ".weight" tensors with >=2
        dimensions (a cheap proxy for "is this a Conv2d/Linear weight").
    """
    errs = []
    for name, w in state_dict.items():
        if not name.endswith("weight") or w.dim() < 2:
            continue
        wq = quantize_tensor(w.float(), bits=bits, per_channel=True, channel_dim=0)
        denom = w.float().norm().clamp_min(1e-12)
        errs.append(((wq - w.float()).norm() / denom).item())
    return sum(errs) / len(errs) if errs else float("nan")


def iter_bit_widths(bits: Iterable[int]) -> List[int]:
    """Normalize/sort a bit-width sweep, always includes 32 (fp32) first if
    present so it is evaluated as a baseline before the quantized variants.
    """
    bits = sorted(set(bits), reverse=True)
    return bits
