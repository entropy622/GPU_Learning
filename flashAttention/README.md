# FlashAttention Sandbox

This directory contains a teaching-oriented baseline for cross-attention:

- `main.cpp`: app entry
- `cpu_cross_attention.*`: CPU reference implementation
- `cuda_cross_attention.*`: CUDA kernel + device runner
- `blockwise_cross_attention.*`: FlashAttention-style blockwise skeleton
- `attention_common.h`: shared config/data structures
- `attention_utils.*`: initialization, validation, reporting
- built-in CUDA event profiling
- correctness check against the CPU output

Build from a VS developer command prompt:

```cmd
cd /d C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
.\flashAttention\build.bat
.\flashAttention\crossAttention.exe
```

Or compile directly with one command:

```cmd
nvcc .\flashAttention\main.cpp .\flashAttention\attention_utils.cpp .\flashAttention\cpu_cross_attention.cpp .\flashAttention\cuda_cross_attention.cu .\flashAttention\blockwise_cross_attention.cu -o .\flashAttention\crossAttention.exe -std=c++17
```

Current implementation is intentionally basic:

- one thread computes one query row
- scores stay in per-thread shared-memory slices for the softmax pass
- no FlashAttention-style online softmax or tiled `K/V` streaming yet

This is meant to be the starting point for:

1. naive cross-attention
2. blockwise / online-softmax cross-attention
3. more realistic FlashAttention-style kernels
