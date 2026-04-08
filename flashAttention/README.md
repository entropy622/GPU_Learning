# FlashAttention 学习目录

这个目录用于从“朴素 cross-attention”逐步走到“FlashAttention 风格的 blockwise 实现”。目标不是直接复刻工业级 kernel，而是把关键算法和工程点一点点落下来，并能结合 profiling 解释为什么这样做。

## 目录结构

- `main.cpp`
  运行入口，支持常规 benchmark 和 profile-only 模式。
- `attention_common.h`
  通用配置与数据结构。
- `attention_utils.*`
  输入初始化、结果校验、输出报告。
- `cpu_cross_attention.*`
  CPU 参考实现，提供正确性基线。
- `cuda_cross_attention.*`
  朴素 CUDA cross-attention baseline。
- `blockwise_cross_attention.*`
  blockwise / FlashAttention-like 实现。
- `online_softmax_notes.md`
  online softmax 与输出累积推导笔记。
- `build.bat`
  一键编译脚本。

## 这部分在学什么

核心是下面这条路径：

1. 先做一个 CPU 版 reference，保证自己知道 attention 在算什么。
2. 再做一个 naive CUDA baseline，建立最基本的 GPU 对照组。
3. 再把 attention 改成 blockwise / tiled 形式：
   - K/V 分块加载
   - shared memory staging
   - online softmax
   - 输出分子与分母同步更新
4. 最后再结合 Nsight Compute 去分析：
   - occupancy
   - warp stall
   - shared memory / global memory 的压力

## 当前实现状态

### 1. CPU reference

`cpu_cross_attention.*` 提供最直白的正确性基线，适合用来验证：

- score 计算是否正确
- softmax 是否稳定
- 输出是否和 GPU 对齐

### 2. naive CUDA baseline

`cuda_cross_attention.*` 采用的是比较直接的做法：

- 一线程对应一个 query row
- 每线程自己计算一整行分数
- 自己做 softmax
- 自己乘 `V`

优点：

- 容易理解
- 适合做第一版 baseline

问题：

- 每线程会缓存完整 score 行
- 当 `keyLen` 变大时，shared memory 需求迅速增大
- 这版实现不适合大规模 profiling

### 3. blockwise / FlashAttention-like 版本

`blockwise_cross_attention.*` 已经实现了这一条关键路线：

- 一个 block 处理一块 query rows
- 沿 K/V 方向按 tile 扫描
- K/V tile 进入 shared memory
- 每线程维护自己的局部 score buffer
- 使用 running max / running sum 做 online softmax
- 输出分子按同一参考系做 rescale 与累积

这版还不是最终高性能 FlashAttention，但算法骨架已经比较接近了。

## 已观察到的性能现象

在较大的输入规模下，blockwise 相比 naive 会有明显加速。

例如在当前配置下的一组实测中：

- `N = 1024` 时，blockwise 相比 naive 约有 `4x` 左右的速度提升

同时，Nsight Compute 已经揭示出当前 blockwise kernel 的一个重要特征：

- 主要 stall 仍然来自访存相关等待，而不是简单的 barrier stall

也就是说，这版实现虽然已经把算法走通了，但在工程性能上还有明显空间，尤其是：

- Q 的缓存方式
- 输出分子的寄存器化
- tile / block 形状选择
- shared memory 与 occupancy 的平衡

## 运行方式

### 常规 benchmark

```cmd
cd /d C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
.\flashAttention\build.bat
.\flashAttention\crossAttention.exe
```

### profile-only 模式

用于 Nsight Compute 时，建议使用 profile-only：

```powershell
$env:FLASH_PROFILE_ONLY="1"
$env:FLASH_QUERY_LEN="8192"
$env:FLASH_KEY_LEN="8192"
$env:FLASH_REPEATS="2"
.\flashAttention\crossAttention.exe
```
