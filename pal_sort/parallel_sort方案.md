# 并行排序方案

目标：在 **整数 / 浮点数的大数组排序** 场景下，尽可能吃满多核 CPU，并且在常见的大规模随机数据上明显超过单线程 `std::sort`。

这份方案同时附带一个可直接编译运行的单文件实现：[parallel_sort.cpp](c:\Users\Aentro\Desktop\Projects\interview\parallel_sort.cpp)。

## 方案选择

我不建议先做并行 quicksort / mergesort，原因很直接：

- `std::sort` 本身已经是非常强的单线程 introsort，想用“同类比较排序”稳定超过它并不容易。
- 比较排序的上界是 `O(n log n)`，而固定宽度数值类型可以用 radix sort 做到 `O(k * n)`。
- radix sort 更容易做成 **分块并行 + 无锁写入不同区间**，对多核 CPU 更友好。

所以这里采用：

**并行 LSD Radix Sort（8 bit / pass）**

适用类型：

- `uint32_t`
- `uint64_t`
- `int32_t`
- `int64_t`
- `float`
- `double`

这也是最现实、最容易直接榨干 CPU 的方案。  
如果你的数据是自定义对象或复杂比较器，这个方案不适合，那时应该改成并行 sample sort / parallel quicksort。

## 核心设计

### 1. 数据分块

把输入数组按连续区间切给多个线程：

- 每个线程只扫描自己那一段
- 避免细粒度任务调度开销
- 对缓存和预取更友好

### 2. 每线程私有直方图

每一轮按 8 bit 做一个 pass，共 `sizeof(T)` 轮：

- 32 位类型做 4 轮
- 64 位类型做 8 轮

每个线程先统计自己的 256 桶直方图：

- 无锁
- 无原子操作
- 减少共享写热点

### 3. 前缀和计算全局写入区间

把所有线程的桶计数汇总后，计算：

- 每个桶的全局起始位置
- 每个线程在每个桶中的起始写入位置

这样每个线程在 scatter 时都写入 **完全不重叠** 的区间。

### 4. 并行 scatter

每个线程再次扫描自己的输入块，把元素写入目标 buffer：

- 不需要锁
- 不需要原子
- 线程之间不会写冲突

每轮 pass 结束后交换 `src/dst buffer`。

## 为什么它有机会超过 `std::sort`

在大规模数值数组上，`std::sort` 的问题主要是：

- 比较次数很多
- 分支预测成本高
- 并行性差

而这个方案的优势是：

- 复杂度更接近线性
- 访问模式规则
- 线程间几乎没有同步热点
- 很容易把性能推到 **内存带宽上限**

在下面这些场景里，通常更容易赢：

- 元素个数很大，至少几百万以上
- 数据类型是 32/64 位整数或浮点
- 数据随机分布较多
- 机器有 8 核、16 核或更多

## 工程判断

这个实现的目标不是“任何场景都赢”，而是：

- **简单可直接上手**
- **代码可直接测试**
- **在目标场景下大概率明显快于单线程 `std::sort`**

需要明确的边界：

- 小数组时，线程和额外内存开销可能让它不如 `std::sort`
- 自定义比较器场景不适用
- 如果你的标准库已经提供并且很好地实现了并行排序，差距会缩小
- 这个版本主要优化的是 **吞吐量**，不是最小内存占用

## 推荐的混合策略

如果你想像 Java 内置排序那样，在不同阶段切不同策略，那么我建议不要坚持“全程只用一种算法”，而是做一个 **自适应混合排序器**。

核心原则：

- 小数组：直接走单线程比较排序
- 中等数组：限制线程数，避免线程管理成本反噬
- 大数组：进入并行 radix 主路径
- 极大数组：复用线程池和临时 buffer，把开销压到最低

建议的决策表如下。

### 1. 小数组策略

当 `n <= 32K`：

- 直接 `std::sort`
- 不开并行
- 不走 radix

原因：

- 此时线程创建、同步、额外内存分配的成本远大于收益
- `std::sort` 对小数组非常强，分支预测和缓存局部性也更好

### 2. 中等数组策略

当 `32K < n <= 1M`：

- 仍然允许 radix
- 但线程数做上限控制
- 建议 `threads = min(hw_threads, max(1, n / 64K))`

原因：

- 中等规模时，盲目全开线程通常会浪费
- 每个线程至少要分到足够多的元素，才能覆盖 histogram 和 scatter 的固定成本

### 3. 大数组策略

当 `n > 1M`：

- 进入并行 radix 主路径
- 默认线程数取 `hardware_concurrency`
- 使用线程池而不是每次排序临时建线程

原因：

- 这个区间已经足以摊平并行开销
- 瓶颈开始更接近内存带宽，而不是算法常数

### 4. 超大数组策略

当 `n > 8M` 或更大：

- 并行 radix
- 复用 scratch buffer
- 复用 histogram / offset buffer
- 按 NUMA 或 cache 拓扑进一步拆分

原因：

- 超大数组下，内存分配和临时对象构造会变得明显
- 此时真正影响性能的通常已经不是“选什么排序”，而是“数据如何流过内存”

## 线程池方案

线程池是你现在最值得先落地的优化项之一。  
因为你当前版本每次 `parallel_for_chunks` 都在反复创建线程，这在中小规模数据上会非常伤。

推荐实现：

- 程序启动时创建固定数量工作线程
- 排序时只提交两个批次任务
- 第一批：histogram
- 第二批：scatter
- 主线程负责 prefix sum 和任务编排

线程池最小功能就够用：

- 固定 worker 数
- 一个任务队列
- 一个简单栅栏或计数器，用于等待本轮任务完成

不需要一开始就做复杂 work-stealing。  
你的任务本身已经是大块连续区间，工作窃取带来的收益通常不如固定分块明显。

### 线程池带来的直接收益

- 去掉每个 pass 的线程创建/销毁
- 小幅降低调度抖动
- 中等规模数组的性能会更稳定
- profile 中 `parallel_region` 的非计算部分会明显下降

## 分阶段排序策略

如果你希望更像 Java 那样“不同阶段实施不同排序策略”，可以按下面设计。

### 阶段 1：入口分流

根据数据类型和规模做第一次决策：

- 固定宽度数值类型：优先 radix
- 自定义对象 / 比较器：走比较排序分支
- 极小数组：直接 `std::sort`

### 阶段 2：块内排序

对于比较排序分支：

- 每个线程先对自己的块做 `std::sort`
- 再做并行 merge

对于 radix 分支：

- 每个线程做本地 histogram
- 再做 prefix sum
- 最后 scatter

### 阶段 3：退化保护

运行时用 profile 结果决定是否降级：

- 如果 `parallel_region` 比例过高，说明线程管理成本太大
- 如果 `copy_in + copy_out` 占比过高，说明额外 buffer 成本太大
- 如果 `scatter` 占比长期极高，说明已接近内存带宽瓶颈

可以加一个简单规则：

- 连续几次排序中，中小数组使用并行版没有赢过 `std::sort`
- 则自动提高并行触发阈值

这就是一个非常实用的 **在线自适应策略**。

## 快速落地的实施顺序

如果目标是尽快交付一个“性能明显更强、结构也合理”的版本，我建议按这个顺序推进。

### 第一步：做自适应入口

先实现一个统一入口：

```cpp
void Sort(int* arr, int len) {
    if (len <= 32 * 1024) {
        std::sort(arr, arr + len);
        return;
    }

    const std::size_t threads = choose_threads(len);
    parallel_radix_sort<int>(std::span<int>(arr, len), threads);
}
```

这是收益最高、改动最小的一步。

### 第二步：引入线程池

把每次排序中的线程创建去掉，改成固定线程池。

这是中等规模数据最关键的优化。

### 第三步：复用临时 buffer

把这些对象做成可复用：

- `scratch`
- `histogram`
- `thread_offsets`

这一步能明显降低分配和初始化成本。

### 第四步：做 profile 驱动阈值调整

现在你已经有 profile 了，可以直接让阈值不再写死：

- `small_threshold`
- `min_elements_per_thread`
- `radix_bits`

可以先离线调，再做在线更新。

## 一个现实可行的最终架构

最现实、最容易短时间落地的版本，不是“一个万能排序器”，而是下面这个结构：

1. `CPUSort::Sort`
2. `choose_strategy(len, type)`
3. `choose_threads(len, hw_threads)`
4. `parallel_radix_sort` 或 `std::sort`
5. 可选 profile 采集
6. 线程池和 buffer 复用

具体策略建议：

- `len <= 32K`：`std::sort`
- `32K < len <= 1M`：并行 radix，但限制线程数
- `len > 1M`：并行 radix + 线程池
- 高频重复调用：并行 radix + 线程池 + buffer 复用 + profile 自适应

## 我最推荐的方案

如果你现在要一个 **最快落地、最容易解释、最容易拿到收益** 的方案，我推荐这一版：

1. 保留 radix 作为大数组主算法
2. 给小数组直接退回 `std::sort`
3. 对中等数组限制线程数
4. 引入固定线程池
5. 复用 scratch / histogram / offsets
6. 用你现在已有的 profile 数据持续调整阈值

这条路线的好处是：

- 代码复杂度可控
- 性能收益很直接
- 很适合面试或工程汇报解释
- 后续还能继续进化到 sample sort / NUMA 优化 / SIMD 优化

## 使用方式

实现文件： [parallel_sort.cpp](c:\Users\Aentro\Desktop\Projects\interview\parallel_sort.cpp)

核心接口：

```cpp
psort::parallel_radix_sort<T>(std::span<T> data, std::size_t threads);
```

例如：

```cpp
std::vector<std::uint64_t> a = ...;
psort::parallel_radix_sort<std::uint64_t>(a, std::thread::hardware_concurrency());
```

## 直接编译测试

### MSVC

```powershell
cl /O2 /std:c++20 /EHsc /MT parallel_sort.cpp
.\parallel_sort.exe u64 20000000 16 3
```

### g++

```powershell
g++ -O3 -march=native -std=c++20 -pthread parallel_sort.cpp -o parallel_sort
.\parallel_sort.exe u64 20000000 16 3
```

### clang++

```powershell
clang++ -O3 -march=native -std=c++20 -pthread parallel_sort.cpp -o parallel_sort
.\parallel_sort.exe u64 20000000 16 3
```

## 参数说明

程序入口：

```text
parallel_sort.exe [u32|u64|i32|i64|f32|f64] [n] [threads] [rounds]
```

示例：

```powershell
.\parallel_sort.exe u64 20000000 16 3
.\parallel_sort.exe i32 50000000 24 3
.\parallel_sort.exe f64 12000000 16 5
```

含义：

- `type`：数据类型
- `n`：元素个数
- `threads`：线程数
- `rounds`：基准测试轮数，程序取 best time

程序会自动做两件事：

- 先校验排序结果是否和 `std::sort` 一致
- 再分别测 `std::sort` 和并行 radix sort 的耗时

## 预期结果

如果机器核数足够、内存带宽不错，并且 `n` 足够大，你应该看到类似结果：

```text
std::sort(best):       1800 ms
parallel_radix(best):   700 ms
speedup:               2.57x
```

实际倍率取决于：

- CPU 核数
- L3 缓存
- 内存带宽
- 编译器优化等级
- 输入数据分布

## 进一步优化方向

如果你后面要继续压性能，可以按这个顺序做：

1. 把每轮线程重复创建，改成固定线程池
2. 直方图做 cache line padding，降低伪共享
3. 每线程使用更大的块内缓冲，减少随机写回
4. 对 `uint32_t` / `uint64_t` 做专门 SIMD 预处理
5. NUMA 机器上按节点分配和归并
6. 区分大数组和超大数组，动态选择 8 bit / 11 bit / 12 bit radix

## 最终建议

如果你的目标是：

- 尽快落地
- 简单上手
- 在数值数组上实打实超过 `std::sort`

那就直接用这份并行 radix sort 方案。

如果你的目标变成：

- 排序对象是结构体
- 需要自定义比较器
- 需要通用模板库级别封装

下一步应切换到 **并行 sample sort**，而不是继续强行扩展 radix。
