# Modified by $@#Anonymous#@$ #20240123
# Copyright (c) 2023, Albert Gu, Tri Dao.
import warnings
import os
from pathlib import Path
from packaging.version import parse, Version
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools import setup

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "FALSE") == "TRUE"

def get_compute_capability():
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    return int(str(capability[0]) + str(capability[1]))
    
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version

_env_modes = os.getenv("SELECTIVE_SCAN_MODES", None)
if _env_modes:
    MODES = [m.strip() for m in _env_modes.split(',') if m.strip()]
else:
    MODES = ["core", "oflex"]  # safe subset present in this repository snapshot

def get_ext():
    cc_flag = []

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    print("\n\nCUDA_HOME = {}\n\n".format(CUDA_HOME))

    # Check if card has compute capability 8.0 or higher for BFloat16 operations
    if get_compute_capability() < 80:
        warnings.warn("This code uses BFloat16 date type, which is only supported on GPU architectures with compute capability 8.0 or higher")
        
    multi_threads = True
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        print("CUDA version: ", bare_metal_version, flush=True)
        if bare_metal_version < Version("11.6"):
            warnings.warn("CUDA version ealier than 11.6 may leads to performance mismatch.")
        if bare_metal_version < Version("11.2"):
            multi_threads = False
            
    cc_flag.append(f"-arch=sm_{get_compute_capability()}")
    
    if multi_threads:
        cc_flag.extend(["--threads", "4"])

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    sources = dict(
        core=[
            "csrc/selective_scan/cus/selective_scan.cpp",
            "csrc/selective_scan/cus/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cus/selective_scan_core_bwd.cu",
        ],
        nrow=[
            "csrc/selective_scan/cusnrow/selective_scan_nrow.cpp",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd2.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd3.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd4.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd2.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd3.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd4.cu",
        ],
        ndstate=[
            "csrc/selective_scan/cusndstate/selective_scan_ndstate.cpp",
            "csrc/selective_scan/cusndstate/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusndstate/selective_scan_core_bwd.cu",
        ],
        oflex=[
            "csrc/selective_scan/cusoflex/selective_scan_oflex.cpp",
            "csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu",
        ],
    )

    names = dict(
        core="selective_scan_cuda_core",
        nrow="selective_scan_cuda_nrow",
        ndstate="selective_scan_cuda_ndstate",
        oflex="selective_scan_cuda_oflex",
    )

    # Filter MODES to only those whose listed source files actually exist.
    valid_modes = []
    for MODE in MODES:
        src_list = sources.get(MODE, [])
        missing = [p for p in src_list if not os.path.exists(os.path.join(this_dir, p))]
        if missing:
            warnings.warn(f"Skip MODE '{MODE}' (missing sources: {missing[:2]}{'...' if len(missing)>2 else ''})")
            continue
        valid_modes.append(MODE)
    if not valid_modes:
        warnings.warn("No valid selective_scan MODES found; no CUDA extensions will be built.")
        return []

    ext_modules = []
    for MODE in valid_modes:
        ext_modules.append(
            CUDAExtension(
                name=names.get(MODE, None),
                sources=sources.get(MODE, None),
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                                "-O3",
                                "-std=c++17",
                                "-U__CUDA_NO_HALF_OPERATORS__",
                                "-U__CUDA_NO_HALF_CONVERSIONS__",
                                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                                "--expt-relaxed-constexpr",
                                "--expt-extended-lambda",
                                "--use_fast_math",
                                "--ptxas-options=-v",
                                "-lineinfo",
                            ]
                            + cc_flag
                },
                include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
            )
        )

    return ext_modules

ext_modules = get_ext()

# Minimal shim: metadata is now in pyproject.toml; keep build_ext so editable/dev installs still compile CUDA.
setup(
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension} if ext_modules else {"bdist_wheel": _bdist_wheel,},
)
