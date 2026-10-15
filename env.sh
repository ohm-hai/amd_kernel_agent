#!/bin/bash
# Primus MoE training optimizations (from AMD ROCm blog:
# https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html)
#
# Source this before launching Primus MoE training:
#   source /root/amd-workspace/env.sh

# Feature 8: CPU Launch Optimization
# NUMA binding: pin each GPU process to its NUMA socket
export ENABLE_NUMA_BINDING=1

# Increase HIP kernel argument pool size to 12MB (avoids launch stalls
# with the many small kernels MoE layers launch)
export HSA_KERNARG_POOL_SIZE=12582912

# Feature: Sync-Free MoE (levels 0-3; 3 = full sync-free, uses more memory)
# export TURBO_SYNC_FREE_MOE_STAGE=3

# Misc used by the MI355X example scripts
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

# Manual GC helper: force periodic host garbage collection to stabilize
# iteration time on long MoE training runs (handled inside Primus training
# scripts, listed here for reference)

# Third-party backend sources for direct (non-container) launches
export PYTHONPATH="/root/amd-workspace/Primus/third_party/torchtitan:/root/amd-workspace/Primus/third_party/Megatron-LM:/root/amd-workspace/Primus/third_party/maxtext:/root/amd-workspace/Primus/third_party/Emerging-Optimizers:/root/amd-workspace/Primus/third_party/Megatron-Bridge:/root/amd-workspace/Primus/third_party/mamba:/root/amd-workspace/Primus/third_party/HummingbirdXT:/root/amd-workspace/Primus/third_party/maxdiffusion:/root/amd-workspace/Primus/third_party/Automodel${PYTHONPATH:+:$PYTHONPATH}"