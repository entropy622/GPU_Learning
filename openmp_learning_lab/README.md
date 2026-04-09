# OpenMP Learning Lab

这个项目是一个独立的 OpenMP 学习实验场，目标不是背指令列表，而是围绕下面这些问题动手：

- 什么是 OpenMP？它解决什么问题？
- Linux 环境下怎么编译和运行？
- 什么地方适合并行，什么地方不适合？
- 什么是数据依赖、数据冲突、原子操作？
- 线程数怎么选？越多越好吗？
- 如何验证并行结果正确？
- 如何计算加速比？
- 如何把 OpenMP 用在矩阵乘法和分块矩阵乘法上？

## OpenMP 是什么

OpenMP 是共享内存并行编程模型。最常见的使用方式是：

1. 用编译指示 `#pragma omp ...` 标记某段代码可并行。
2. 用支持 OpenMP 的编译器打开开关编译。
3. 在一个进程内启动多个线程并行执行循环或代码块。

它特别适合：

- CPU 多核上的循环并行
- 共享内存机器上的数值计算
- 把串行 baseline 快速改成一个可验证的并行版本

它不适合直接解决：

- 多机分布式内存问题
- 强数据依赖、迭代之间必须严格串行的算法

## 目录

- `src/omp_basics.cpp`
  OpenMP 基本概念：parallel region、data race、atomic、reduction、schedule。
- `src/omp_thread_sweep.cpp`
  扫描线程数，观察“线程越多未必越快”。
- `src/omp_matrix_mul.cpp`
  串行矩阵乘法、外层循环并行、`collapse(2)` 对比。
- `src/omp_blocked_matmul_exercise.cpp`
  分块矩阵乘法练习版。保留关键 TODO，留给你亲自完成。

## Linux 下如何配置

### GCC

大多数 Linux 环境下直接装 `g++` 就能得到 OpenMP 支持：

```bash
sudo apt update
sudo apt install build-essential cmake
```

编译时加：

```bash
g++ -O2 -fopenmp your_file.cpp -o your_program
```

### Clang

Clang 也能用 OpenMP，但很多发行版还需要安装运行时：

```bash
sudo apt install clang libomp-dev cmake
```

编译方式会因发行版略有区别，最稳的是先用 GCC 起步。

### 运行时控制线程数

Linux 下常用环境变量：

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

最常用的是 `OMP_NUM_THREADS`。

## Windows 下当前仓库的构建方式

```powershell
cd C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
cmake -S .\openmp_learning_lab -B .\openmp_learning_lab\build -G "MinGW Makefiles"
cmake --build .\openmp_learning_lab\build
```

或者：

```powershell
.\openmp_learning_lab\build.bat
```

## 我该在哪并行

优先找这些模式：

- 大循环，迭代次数多
- 每次迭代工作量大致均匀
- 每次迭代写不同的输出元素
- 迭代之间没有强依赖

矩阵乘法里，外层 `i` 或 `(i, j)` 通常适合并行，因为不同线程写不同的 `C[i][j]`。

不适合直接并行的地方：

- 一个迭代依赖前一个迭代结果
- 多个线程同时写同一个标量或数组元素
- 工作量极小，线程调度开销大于收益

## 什么是数据依赖、数据冲突

### 数据依赖

如果第 `i` 次迭代需要用到第 `i-1` 次迭代的结果，就有依赖。例如前缀和。

这种循环不能直接无脑 `parallel for`。

### 数据冲突

如果多个线程同时写同一个共享变量，就可能冲突。例如：

```cpp
int sum = 0;
#pragma omp parallel for
for (...) {
  sum += 1;
}
```

这会 data race。

常见解决方式：

- `reduction`
- `atomic`
- `critical`
- 改写算法，让每个线程先写本地结果，再汇总

## 什么是原子操作，为什么需要

OpenMP 中常见形式：

```cpp
#pragma omp atomic
++counter;
```

它的含义是：

- 这次更新必须是原子的
- 避免多个线程同时更新同一个变量导致结果错误

但要记住：

- `atomic` 解决正确性
- 不保证高性能
- 热点原子会形成串行瓶颈

如果是求和、计数，优先想想能不能用 `reduction`。

## 线程数量怎么选

不要假设“越多越快”。

线程数增加后可能遇到：

- 核心数不够
- 超线程收益有限
- 内存带宽饱和
- 调度开销增加
- 竞争和同步开销增加

所以要测。`omp_thread_sweep` 就是专门给你做这件事的。

## 如何验证正确性

做 OpenMP 优化时，顺序应该永远是：

1. 先写串行 reference
2. 再写并行版本
3. 每次优化后都和 reference 对比

本项目里矩阵乘法都保留了串行版本，并用 `almost_equal` 做结果比较。

## 如何计算加速比

公式：

```text
speedup = serial_time / parallel_time
```

例如串行 100 ms，并行 40 ms：

```text
speedup = 100 / 40 = 2.5
```

注意：

- 要在相同输入规模下比较
- 最好跑多次取稳定值
- 正确性没验证前，不要讨论性能

## 建议学习顺序

1. 先跑 `omp_basics`
2. 再跑 `omp_thread_sweep`
3. 再跑 `omp_matrix_mul`
4. 最后补完 `omp_blocked_matmul_exercise`

## 运行示例

```powershell
.\openmp_learning_lab\build\omp_basics.exe
.\openmp_learning_lab\build\omp_thread_sweep.exe
.\openmp_learning_lab\build\omp_matrix_mul.exe 256
.\openmp_learning_lab\build\omp_blocked_matmul_exercise.exe 256 32
```

Linux 下：

```bash
./omp_basics
OMP_NUM_THREADS=8 ./omp_thread_sweep
OMP_NUM_THREADS=8 ./omp_matrix_mul 256
OMP_NUM_THREADS=8 ./omp_blocked_matmul_exercise 256 32
```

## 每个文件学什么

### `omp_basics.cpp`

学这些：

- `#pragma omp parallel`
- `#pragma omp parallel for`
- `#pragma omp atomic`
- `reduction`
- 为什么会 data race

### `omp_thread_sweep.cpp`

学这些：

- 线程数调优必须靠测量
- 加速曲线通常不是线性的
- 线程太多可能反而退化

### `omp_matrix_mul.cpp`

学这些：

- 为什么外层循环适合并行
- `collapse(2)` 在二维循环上的意义
- 正确性验证和加速比计算

### `omp_blocked_matmul_exercise.cpp`

你要亲手完成：

- 选择并行区域
- 在正确位置加 OpenMP pragma
- 把朴素乘法改成真正的 blocked matmul
- 比较 block size 和线程数的影响

## 你应该能回答的面试问题

- OpenMP 是什么？适合什么硬件模型？
- `parallel for` 应该加在哪一层循环？
- 什么是 data race？怎么修？
- `atomic` 和 `reduction` 有什么区别？
- 为什么线程数更多不一定更快？
- 如何验证并行程序正确？
- 如何定义和计算加速比？
- 为什么矩阵乘法适合外层并行和 cache blocking？
