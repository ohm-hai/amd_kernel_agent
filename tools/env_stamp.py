"""Environment stamp for kernel-lab evidence records.

Every benchmark, verifier and profiling result should carry `env_stamp()` so it can be
reproduced and so results from a busy GPU can be discounted. Usage:

    from env_stamp import env_stamp, require_free_gpu
    require_free_gpu()            # raises if the selected GPU has other tenants
    stamp = env_stamp()           # dict: versions, physical GPU, occupancy, time

CLI: `python3 tools/env_stamp.py` prints the stamp for HIP_VISIBLE_DEVICES.
"""
import datetime
import importlib.metadata as md
import json
import os
import socket
import subprocess

BUSY_VRAM_PCT = 5   # other tenants usually hold far more; our own runs start near 0
BUSY_USE_PCT = 15


def _version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _rocm_version():
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        if os.path.exists(path):
            return open(path).read().strip()
    return None


def physical_gpu():
    """Physical GPU index selected by HIP_VISIBLE_DEVICES (first entry), or 0."""
    vis = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    return int(vis.split(",")[0]) if vis else 0


def gpu_occupancy(gpu):
    """VRAM% and GPU use% for a physical GPU from rocm-smi, or None if unavailable."""
    try:
        out = subprocess.run(["rocm-smi", "--showmemuse", "--showuse", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        card = json.loads(out).get(f"card{gpu}", {})
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    vram = card.get("GPU Memory Allocated (VRAM%)")
    use = card.get("GPU use (%)")
    return {"vram_pct": float(vram) if vram is not None else None,
            "use_pct": float(use) if use is not None else None}


def env_stamp():
    gpu = physical_gpu()
    stamp = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "physical_gpu": gpu,
        "gpu_occupancy_at_start": gpu_occupancy(gpu),
        "rocm": _rocm_version(),
        "torch": _version("torch"),
        "pytorch_triton_rocm": _version("pytorch-triton-rocm"),
        "flash_linear_attention": _version("flash-linear-attention"),
        "flydsl": _version("flydsl"),
        "tracelens": _version("TraceLens"),
    }
    try:
        import torch
        stamp["hip_runtime"] = torch.version.hip
        if torch.cuda.is_available():
            stamp["gcn_arch"] = torch.cuda.get_device_properties(0).gcnArchName
    except ImportError:
        pass
    return stamp


def require_free_gpu(gpu=None):
    """Raise if the GPU already carries another tenant's memory or load."""
    gpu = physical_gpu() if gpu is None else gpu
    occ = gpu_occupancy(gpu)
    if occ is None:
        return
    if (occ["vram_pct"] or 0) > BUSY_VRAM_PCT or (occ["use_pct"] or 0) > BUSY_USE_PCT:
        raise RuntimeError(
            f"GPU {gpu} is busy (VRAM {occ['vram_pct']}%, use {occ['use_pct']}%). "
            f"Pick a free GPU with `rocm-smi` and set HIP_VISIBLE_DEVICES.")


if __name__ == "__main__":
    print(json.dumps(env_stamp(), indent=2))
