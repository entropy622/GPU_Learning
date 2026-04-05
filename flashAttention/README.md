# FlashAttention Sandbox

This directory contains a teaching-oriented baseline for cross-attention:

- `crossAttention.cu`: CPU reference + simple CUDA cross-attention kernel
- built-in CUDA event profiling
- correctness check against the CPU output

Build from a VS developer command prompt:

```cmd
cd /d C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
nvcc .\flashAttention\crossAttention.cu -o .\flashAttention\crossAttention.exe -std=c++17
.\flashAttention\crossAttention.exe
```

Current implementation is intentionally basic:

- one thread computes one query row
- scores stay in per-thread shared-memory slices for the softmax pass
- no FlashAttention-style online softmax or tiled `K/V` streaming yet

This is meant to be the starting point for:

1. naive cross-attention
2. blockwise / online-softmax cross-attention
3. more realistic FlashAttention-style kernels
