# GPU_Learning

这是一个围绕 CUDA 基础、矩阵乘优化、Attention / FlashAttention 思路展开的学习型项目。

## 项目结构

- `cuda_basic/`
  CUDA 入门与基础优化练习。
- `flashAttention/`
  面向 FlashAttention 学习路径的 cross-attention 与 blockwise attention 实验。
- `.vscode/`
  VS Code 的 IntelliSense 配置。

## 当前内容

### 1. CUDA 基础部分

位于 `cuda_basic/`：

- `vecAdd.cu`
  最基础的 CUDA 向量加法，用来理解线程索引和 kernel launch。
- `naiveMatmul.cu`
  朴素矩阵乘，核心是“一线程对应一个输出元素”。
- `tileMatmul.cu`
  使用 shared memory 做 tiled matmul，对比 naive 版本并测时间。

这一部分主要解决：

- thread/block/grid 的映射方式
- global memory 与 shared memory 的角色
- 为什么 tiled matmul 会比 naive matmul 更快
- 如何用 CUDA events 和 Nsight Compute 初步分析性能

### 2. FlashAttention 学习部分

位于 `flashAttention/`：

- `cpu_cross_attention.*`
  CPU 参考实现，作为正确性基线。
- `cuda_cross_attention.*`
  朴素 CUDA cross-attention 基线实现。
- `blockwise_cross_attention.*`
  逐步搭建出来的 blockwise / FlashAttention-like 实现。
- `online_softmax_notes.md`
  online softmax 与输出累积推导笔记。

这一部分主要解决：

- cross-attention 的基本实现
- 为什么 attention 的优化重点常常是 IO，不只是 FLOPs
- 为什么 FlashAttention 需要 online softmax
- blockwise 扫描 K/V tile 时，running max / running sum / output accumulator 如何更新

## 环境

当前项目在 Windows + CUDA Toolkit + Visual Studio 开发者命令行环境下使用。

已验证的工具链包括：

- NVIDIA CUDA Toolkit
- `nvcc`
- Visual Studio 2022 Community
- VS Code
- Nsight Compute

## 编译方式

### CUDA 基础示例

在 VS 开发者命令行中：

```cmd
cd /d C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
nvcc .\cuda_basic\vecAdd.cu -o .\cuda_basic\vecAdd.exe -std=c++17
nvcc .\cuda_basic\naiveMatmul.cu -o .\cuda_basic\naiveMatmul.exe -std=c++17
nvcc .\cuda_basic\tileMatmul.cu -o .\cuda_basic\tileMatmul.exe -std=c++17
```

### FlashAttention 目录

```cmd
cd /d C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
.\flashAttention\build.bat
.\flashAttention\crossAttention.exe
```

## Profiling

项目里已经支持一个更干净的 profile-only 模式，主要用于 Nsight Compute：

```powershell
$env:FLASH_PROFILE_ONLY="1"
$env:FLASH_QUERY_LEN="8192"
$env:FLASH_KEY_LEN="8192"
$env:FLASH_REPEATS="2"
.\flashAttention\crossAttention.exe
```

配合 `ncu`：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\ncu.bat" `
  --target-processes all `
  --kernel-name-base demangled `
  --kernel-name "<unnamed>::blockwiseAttentionSkeletonKernel(const float *, const float *, const float *, float *, int, int, int, int, int, int, float)" `
  --section LaunchStats `
  --section Occupancy `
  --section WarpStateStats `
  --section SchedulerStats `
  .\flashAttention\crossAttention.exe
```