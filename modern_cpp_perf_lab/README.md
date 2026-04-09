# Modern C++ Performance Lab

这个子项目不是语法抄写本，而是一个面向高性能岗位面试的练习场。目标是把下面几件事练到能解释、能写、能分析：

- RAII 和资源生命周期
- 拷贝/移动语义与接口设计
- `std::thread`、`std::mutex`、`std::atomic`
- `memory_order` 的最小必要理解
- MPI 的进程模型与基础通信原语

## 目录

- `src/raii_demo.cpp`
  通过缓冲区、文件和异常路径演示 RAII 的确定性清理。
- `src/move_semantics_demo.cpp`
  用带计数的类型观察 copy/move、`push_back`、`emplace_back`、`reserve`。
- `src/concurrency_counter_demo.cpp`
  对比 `mutex`、热点 `atomic fetch_add`、分片局部计数器。
- `src/memory_order_demo.cpp`
  演示 `relaxed`、`release/acquire`、`seq_cst` 的典型使用场景。
- `src/mpi_basics_demo.cpp`
  MPI 可选示例，覆盖 `rank/size`、`Send/Recv`、`Bcast`、`Scatter/Gather`、`Reduce/Allreduce`。

## 为什么这样设计

### 1. RAII

面试里常见误区是把 RAII 等同于智能指针。实际上文件句柄、锁、socket、显存句柄都应该用对象生命周期管理。你要能回答：

- 为什么异常路径不应该手写 `cleanup`。
- 为什么析构函数让资源释放和控制流解耦。
- 为什么高性能代码更需要明确 ownership，而不是更少管理。

### 2. Move 语义

高性能岗位不会只问“什么是右值引用”，而是更关心：

- 这个接口会不会制造不必要临时对象。
- `std::vector` 扩容时你的类型是复制还是移动。
- 返回对象时是否阻碍 NRVO/移动。
- `reserve` 是否减少搬运成本。

### 3. 原子和并发

你至少要区分三件事：

- 正确性：普通共享读写会 data race，原子让行为有定义。
- 性能：原子不等于快，热点原子可能成为串行瓶颈。
- 同步语义：`relaxed`、`release/acquire`、`seq_cst` 解决的问题不同。

### 4. MPI

MPI 看的是你是否真的明白“多进程 + 分布式内存”：

- 每个 rank 默认有自己的地址空间。
- 跨 rank 交换数据靠消息传递，不是共享变量。
- collectives 的语义和参与者要求要清楚。
- 多节点性能不稳定，通常先怀疑网络、拓扑和环境一致性。

## 构建

### Windows + MinGW / clang++

```powershell
cd C:\Users\Aentro\Desktop\Projects\github\GPU_Learning
cmake -S .\modern_cpp_perf_lab -B .\modern_cpp_perf_lab\build -G "MinGW Makefiles"
cmake --build .\modern_cpp_perf_lab\build
```

### 运行

```powershell
.\modern_cpp_perf_lab\build\raii_demo.exe
.\modern_cpp_perf_lab\build\move_semantics_demo.exe
.\modern_cpp_perf_lab\build\concurrency_counter_demo.exe
.\modern_cpp_perf_lab\build\memory_order_demo.exe
```

## MPI 构建

当前 CMake 会自动查找 MPI。安装 Open MPI 或 MS-MPI 后重新配置即可。如果本机没有 MPI，主工程仍然可以正常构建。

示例运行方式：

```powershell
mpiexec -n 4 .\modern_cpp_perf_lab\build\mpi_basics_demo.exe
```

## 建议学习顺序

1. 先跑 `raii_demo`，把“资源获取即初始化”和异常路径彻底吃透。
2. 再跑 `move_semantics_demo`，重点看 `reserve` 和 `std::move` 前后 copy/move 计数变化。
3. 跑 `concurrency_counter_demo`，理解“atomic 正确但可能更慢”。
4. 跑 `memory_order_demo`，把“统计”和“发布数据”两类场景拆开记。
5. 最后装 MPI，跑 `mpi_basics_demo`，把 point-to-point 和 collective 的语义串起来。

## 面试回答模板

### 原子类型

“如果只是计数，我先考虑 `memory_order_relaxed`；如果一个线程构造了数据，另一个线程要在看到标志位后读取它，我会用 `release`/`acquire`；如果我先要保证正确性、代码还在验证阶段，我会先用 `seq_cst`，再看有没有必要放松。”

### 高性能接口设计

“我会先看这个接口是否引入不必要拷贝、是否阻碍对象移动、是否让 `vector` 扩容成本放大、是否无意中制造临时对象，以及共享状态是否被一个热点锁或热点原子串行化。”

## 你下一步可以继续扩展

- 加一个简化线程池，练任务队列与 RAII 的结合。
- 加一个矩阵乘法 CPU baseline，观察连续存储和缓存命中。
- 加一个 false sharing 对比实验，理解 cache line 竞争。
- 把 MPI demo 扩成 block matrix multiply，和你后面的 CUDA/MPI 混合场景接起来。
