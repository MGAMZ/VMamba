"""Unified high-level API for selective_scan CUDA kernels.

This wrapper hides the multiple low-level extension module names used in the
repo (``selective_scan_cuda_oflex``, ``selective_scan_cuda_core``,
``selective_scan_cuda``) and provides a single function ``selective_scan``.

Typical minimal usage (auto backend selection, returns output only):

    from vmamba.kernels.selective_scan import selective_scan
    out = selective_scan(u, delta, A, B, C)

If you also need the last recurrent state:

    out, last_state = selective_scan(u, delta, A, B, C, return_last_state=True)

Arguments (canonical shapes):
    u:          Float tensor (B, D, L) or (B, K*C, L).
                Input sequence features grouped implicitly into K groups each of size C.
    delta:      Float tensor (B, D, L). Step sizes / decay (will softplus if activation="softplus").
    A:          Float tensor (D, N).   Log state transition coefficients (stored already in required form).
    B:          Float tensor (B, K, N, L) or (B, D, N, L) or broadcastable variant.
                Input projection over sequence. If provided as (B, K, N, L) we expand along C automatically.
    C:          Same shape rules as B. Output projection.
    D:          Optional float tensor (D,) skip/residual term.
    delta_bias: Optional float tensor (D,) bias added to delta before activation.

Keyword Options:
    backend:    "auto" (default) | "oflex" | "core" | "mamba" | "torch".
                "torch" forces the pure PyTorch fallback (slow, always available).
                Others require corresponding compiled extension; will fall back automatically.
    activation: "softplus" (default) or "none" (bypass non-linearity on delta).
    oflex_float_out: bool (default True). When using oflex kernel, convert output back to input dtype.
    return_last_state: bool. If True returns (out, last_state) where last_state is (B, D, N).
    nrows / backnrows: internal tiling hints for oflex kernel (1..4). Usually leave default.

Returns:
    out or (out, last_state) with out shape (B, D, L) matching input dtype.

Error Modes:
    - Shape mismatch: raises AssertionError with a short message.
    - Missing desired backend: silently falls back to available one unless backend explicitly set.

Performance Notes:
    - Prefer backend="oflex" (fastest) if compiled; else "mamba" / "core"; else fallback to pure torch.
    - Mixed precision: inputs may be half / bfloat16; internal math may upcast to float32 for stability.

This API purposefully keeps parameter naming identical to existing internal code
so migration is trivial.
"""
from __future__ import annotations
import importlib
import warnings
from typing import Optional, Tuple, Union
import torch


def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None

_EXT_OFLEX = _try_import("selective_scan_cuda_oflex")
_EXT_CORE = _try_import("selective_scan_cuda_core")
_EXT_MAMBA = _try_import("selective_scan_cuda")  # original mamba kernel name

def _expand_BC(B: torch.Tensor, C: torch.Tensor, D: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Expand (B, K, N, L) -> (B, D, N, L) if needed."""
    if B.dim() == 4 and B.shape[1] != D:
        K = B.shape[1]
        assert D % K == 0, "D must be divisible by groups (K) for expansion"
        cdim = D // K
        B = B.view(B.shape[0], K, 1, B.shape[2], B.shape[3]).repeat(1, 1, cdim, 1, 1).view(B.shape[0], D, B.shape[2], B.shape[3])
        C = C.view(C.shape[0], K, 1, C.shape[2], C.shape[3]).repeat(1, 1, cdim, 1, 1).view(C.shape[0], D, C.shape[2], C.shape[3])
        return B, C, K
    return B, C, -1

def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    *,
    backend: str = "auto",
    activation: str = "softplus",
    return_last_state: bool = False,
    nrows: int = 1,
    backnrows: int = -1,
    oflex_float_out: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert u.dim() == 3, "u must be (B, D, L)"
    assert delta.shape == u.shape, "delta must match u shape"
    assert A.dim() == 2, "A must be (D, N)"
    assert B.dim() == 4 and C.dim() == 4, "B and C must be 4D (B, K|D, N, L)"
    B_, Dtot, L = u.shape[0], u.shape[1], u.shape[2]
    assert A.shape[0] == Dtot, "A first dim must equal D"
    N = A.shape[1]
    B, C, K = _expand_BC(B, C, Dtot)
    assert B.shape == (B_, Dtot, N, L)
    assert C.shape == (B_, Dtot, N, L)
    if delta_bias is not None:
        assert delta_bias.shape[0] == Dtot, "delta_bias must be (D,)"
    if activation == "softplus":
        delta_eff = torch.nn.functional.softplus(delta + (delta_bias[..., None] if delta_bias is not None else 0))
    elif activation == "none":
        delta_eff = delta + (delta_bias[..., None] if delta_bias is not None else 0)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    want = backend.lower()
    chosen = None
    ext = None
    if want == "torch":
        chosen = "torch"
    elif want == "oflex" and _EXT_OFLEX is not None:
        chosen = "oflex"; ext = _EXT_OFLEX
    elif want == "core" and _EXT_CORE is not None:
        chosen = "core"; ext = _EXT_CORE
    elif want == "mamba" and _EXT_MAMBA is not None:
        chosen = "mamba"; ext = _EXT_MAMBA
    elif want == "auto":
        if _EXT_OFLEX is not None:
            chosen = "oflex"; ext = _EXT_OFLEX
        elif _EXT_CORE is not None:
            chosen = "core"; ext = _EXT_CORE
        elif _EXT_MAMBA is not None:
            chosen = "mamba"; ext = _EXT_MAMBA
        else:
            chosen = "torch"
    else:
        warnings.warn(f"Requested backend '{backend}' not available; falling back to 'torch'.")
        chosen = "torch"
    if chosen == "torch":
        dtype_in = u.dtype
        u_f = u.float(); delta_f = delta_eff.float(); A_f = A.float(); B_f = B.float(); C_f = C.float()
        x = A_f.new_zeros((B_, Dtot, N))
        ys = []
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta_f, A_f))
        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta_f, B_f, u_f)
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C_f[:, :, :, i])
            ys.append(y)
        out = torch.stack(ys, dim=2)
        if D is not None:
            out = out + u_f * D.unsqueeze(-1).float()
        out = out.to(dtype_in)
        if return_last_state:
            return out, x
        return out
    D_arg = D if D is not None else None
    delta_bias_arg = delta_bias if delta_bias is not None else None
    if chosen == "oflex":
        out, x, *_ = ext.fwd(u, delta_eff, A, B, C, D_arg, delta_bias_arg, False, nrows, oflex_float_out)
    elif chosen == "core":
        out, x, *_ = ext.fwd(u, delta_eff, A, B, C, D_arg, delta_bias_arg, False, nrows)
    elif chosen == "mamba":
        out, x, *_ = ext.fwd(u, delta_eff, A, B, C, D_arg, None, delta_bias_arg, False)
    else:
        raise RuntimeError(f"Unexpected backend chosen={chosen}")
    if return_last_state:
        last_state = x[:, :, -1, 1::2] if x.dim() == 4 else x
        return out.to(u.dtype), last_state
    return out.to(u.dtype)


from .test_selective_scan import selective_scan_ref, selective_scan_ref_v2, selective_scan_fn

__all__ = ["selective_scan",
           "selective_scan_fn",
           "selective_scan_ref",
           "selective_scan_ref_v2"]
