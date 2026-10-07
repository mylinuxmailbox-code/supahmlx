#include <metal_stdlib>
using namespace metal;

// Phase 1 baseline kernels/placeholders for experimentation only.
// TODO: benchmark threadgroup memory vs direct global loads per op.
// TODO: evaluate SIMD-group reductions for RMSNorm and attention softmax.
// TODO: sweep tile sizes and occupancy for matmul/attention kernels.
// TODO: evaluate memory layout changes (AoS/SoA/interleaved) for KV-cache reads.

kernel void rmsnorm_baseline(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& length [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= length) {
        return;
    }
    output[gid] = input[gid];
}

kernel void rope_baseline(device const float2* input [[buffer(0)]],
                          device float2* output [[buffer(1)]],
                          constant uint& length [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= length) {
        return;
    }
    output[gid] = input[gid];
}

kernel void fused_activation_baseline(device const float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant uint& length [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid >= length) {
        return;
    }
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x)); // SiLU baseline
}

kernel void tiled_matmul_placeholder(device const float* a [[buffer(0)]],
                                     device const float* b [[buffer(1)]],
                                     device float* c [[buffer(2)]],
                                     constant uint& n [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    c[gid] = a[gid] * b[gid];
}
