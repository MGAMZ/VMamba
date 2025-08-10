"""Auxiliary setup script to build the selective_scan CUDA extension automatically.

We keep metadata in pyproject.toml; this file only injects ext_modules when
`pip install .` is executed. If torch / CUDA aren't available, we skip the build
gracefully so the pure-python parts remain usable (will fall back to slower torch impls).

Environment variables:
  VMAMBA_SKIP_EXT=1          -> Skip building any native extensions.
  VMAMBA_BUILD_SELECTIVE_SCAN=0 -> Same effect as skip for this specific ext.

You can later build the extension in-place via:
  python setup.py build_ext --inplace
or reinstall with the env vars unset.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from setuptools import setup

def should_build() -> bool:
    if os.environ.get("VMAMBA_SKIP_EXT") == "1":
        return False
    if os.environ.get("VMAMBA_BUILD_SELECTIVE_SCAN") == "0":
        return False
    return True

def get_ext_modules():  # pragma: no cover - build helper
    if not should_build():
        print("[vmamba] Skipping selective_scan extension (env var).", flush=True)
        return []
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension
    except Exception as e:  # noqa: BLE001
        print(f"[vmamba] Torch not available, skip building selective_scan extension: {e}", flush=True)
        return []
    if not torch.cuda.is_available():
        print("[vmamba] CUDA not available at build time; skipping selective_scan extension.", flush=True)
        return []
    this_dir = Path(__file__).parent.resolve()
    ss_root = this_dir / "kernels" / "selective_scan"
    csrc = ss_root / "csrc" / "selective_scan"
    sources = [
        csrc / "cusoflex" / "selective_scan_oflex.cpp",
        csrc / "cusoflex" / "selective_scan_core_fwd.cu",
        csrc / "cusoflex" / "selective_scan_core_bwd.cu",
    ]
    sources = [str(s) for s in sources]
    extra_nvcc = [
        "-O3", "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v",
    ]
    try:
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{cc_major}{cc_minor}"
        extra_nvcc.append(f"-arch={arch}")
    except Exception as e:  # noqa: BLE001
        print(f"[vmamba] Could not get device capability ({e}); using generic arch list.", flush=True)
        extra_nvcc.extend(["-arch=sm_80"])  # baseline for Ampere+
    ext = CUDAExtension(
        name="selective_scan_cuda_oflex",
        sources=sources,
        include_dirs=[str(csrc)],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": extra_nvcc,
        },
    )
    print("[vmamba] Prepared selective_scan_cuda_oflex extension (sources count=", len(sources), ")", flush=True)
    return [ext]

if __name__ == "__main__":  # pragma: no cover
    setup(ext_modules=get_ext_modules())
else:
    setup(ext_modules=get_ext_modules())
