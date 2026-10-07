# Metal Kernel Work Plan

`src/supahmlx/metal/kernels/baseline_kernels.metal` contains safe baseline kernels/placeholders for:

- RMSNorm baseline path
- RoPE baseline path
- fused activation baseline
- tiled matmul placeholder

These are **not production optimized kernels**.

Planned experiments:

- threadgroup-memory tiling
- SIMD-group reductions
- tile-size/occupancy sweeps
- memory layout experiments for KV cache
- launch/synchronization overhead profiling
