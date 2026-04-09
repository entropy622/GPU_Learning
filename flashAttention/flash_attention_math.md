# 这一版 FlashAttention 的数学原理

这份笔记专门解释当前 `flashAttention/` 目录里这版 **blockwise / FlashAttention-like** 实现背后的数学逻辑。  
目标不是介绍论文里所有工程细节，而是把当前代码里真正已经落地的数学链条讲清楚。

对应实现文件主要是：

- `flashAttention/blockwise_cross_attention.cu`
- `flashAttention/online_softmax_notes.md`

---

## 1. 我们到底在算什么

当前目录实现的是单头、单 batch 的 cross-attention。

给定：

$$
Q \in \mathbb{R}^{N_q \times d}
$$

$$
K \in \mathbb{R}^{N_k \times d}
$$

$$
V \in \mathbb{R}^{N_k \times d_v}
$$

标准 attention 输出为：

$$
S = \frac{QK^T}{\sqrt{d}}
$$

$$
P = \text{softmax}(S)
$$

$$
O = PV
$$

对于某一个 query 行 $q_i$，它的输出是：

$$
o_i = \sum_{j=1}^{N_k} \text{softmax}(s_{ij}) \, v_j
$$

其中：

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}
$$

---

## 2. 标准实现为什么不适合大规模

如果直接按定义计算，最自然的方法是：

1. 先算出整块 score matrix

$$
S \in \mathbb{R}^{N_q \times N_k}
$$

2. 对每一行做 softmax
3. 再乘上 $V$

问题在于：

- 中间矩阵 $S$ 很大
- 如果显式物化 $S$，会带来大量 global memory / HBM 读写
- 真正的瓶颈常常不是 FLOPs，而是 IO

FlashAttention 类方法的核心思想就是：

> 不显式存完整 attention matrix，而是按块计算、按块归一化、按块累积输出。

---

## 3. 当前代码的 blockwise 思想

当前 `blockwise_cross_attention.cu` 采用的是这样的思路：

1. 一次处理一小块 query rows
2. 沿着 K/V 方向一块一块扫描
3. 当前轮把一小块 `K tile` 和 `V tile` 搬到 shared memory
4. 对当前 query row，计算当前 tile 的局部 score
5. 用 online softmax 把“旧 tile”和“新 tile”的统计量拼起来
6. 同步更新输出向量

也就是说，当前不是：

$$
\text{先算完整 } S，再算 softmax，再乘 V
$$

而是：

$$
\text{每次只看一个 } K/V \text{ tile，边扫边更新}
$$

---

## 4. 对单个 query row 的重写

固定某个 query row $q_i$，标准 attention 输出可以写成：

$$
o_i = \frac{\sum_{j=1}^{N_k} e^{s_{ij}} v_j}{\sum_{j=1}^{N_k} e^{s_{ij}}}
$$

这说明 attention 输出本质上是：

$$
o_i = \frac{\text{分子}}{\text{分母}}
$$

其中分母是：

$$
\ell = \sum_{j=1}^{N_k} e^{s_{ij}}
$$

分子是：

$$
a = \sum_{j=1}^{N_k} e^{s_{ij}} v_j
$$

所以只要我们能边扫描边维护这两个量：

- 标量分母 $\ell$
- 向量分子 $a$

最后就能恢复输出：

$$
o_i = \frac{a}{\ell}
$$

---

## 5. 为什么 softmax 里会出现 max

softmax 定义里本来没有 max：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

但 softmax 有一个重要性质：

$$
\text{softmax}(x_i) = \text{softmax}(x_i - c)
$$

对整行减去同一个常数 $c$，结果不变。

于是我们选：

$$
c = \max_j x_j
$$

得到更稳定的形式：

$$
\text{softmax}(x_i)
=
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
\quad\text{其中}\quad
m = \max_j x_j
$$

这样所有指数输入都不会太大，因为：

$$
x_i - m \le 0
$$

---

## 6. 为什么 blockwise 需要 online softmax

如果是整行一次算完，那么：

- 整行最大值 $m$
- 整行分母

都可以一次得到。

但当前实现不是整行一次算完，而是把 key 方向拆成若干个 tile：

$$
\text{tile}_1, \text{tile}_2, \dots, \text{tile}_T
$$

所以我们在处理第 $t$ 块时，并不知道最终全局最大值。  
这就是为什么必须维护“运行时状态”：

- `runningMax`
- `runningSum`

在代码里，它们就是：

```cpp
float runningMax = -FLT_MAX;
float runningSum = 0.0f;
```

---

## 7. runningMax 与 runningSum 的数学含义

处理完若干个 tile 后，定义：

$$
m_{\text{prev}} = \text{runningMax}
$$

$$
\ell_{\text{prev}} = \text{runningSum}
$$

其中 $\ell_{\text{prev}}$ 的真正含义不是原始分母，而是：

$$
\ell_{\text{prev}} = \sum_{\text{old } j} e^{s_{ij} - m_{\text{prev}}}
$$

也就是说：

- `runningMax` 是到目前为止的全局最大值
- `runningSum` 是“相对于这个最大值归一化之后”的指数和

这就是 online softmax 的核心状态表示。

---

## 8. 当前 tile 的局部计算

对于当前 tile，代码先计算：

$$
s_{i, t, 1}, s_{i, t, 2}, \dots, s_{i, t, B}
$$

其中 $B = \text{tileCols}$。

对应代码里：

```cpp
float score = 0.0f;
for (int dim = 0; dim < headDim; ++dim) {
    score += q[queryIdx * headDim + dim] * keyTile[tileCol * headDim + dim];
}
score *= scale;
localScores[tileCol] = score;
```

然后取当前 tile 最大值：

$$
m_{\text{tile}} = \max(s_{i,t,1}, \dots, s_{i,t,B})
$$

对应：

```cpp
tileMax = fmaxf(tileMax, score);
```

---

## 9. 为什么需要重新缩放旧状态

新的全局最大值应该是：

$$
m_{\text{new}} = \max(m_{\text{prev}}, m_{\text{tile}})
$$

这对应代码：

```cpp
float newMax = fmaxf(runningMax, tileMax);
```

现在问题来了：

旧的 `runningSum` 是按 $m_{\text{prev}}$ 表示的，而新 tile 想按 $m_{\text{new}}$ 表示。  
要把它们加起来，必须先换到同一个参考系。

利用：

$$
e^{x - m_{\text{new}}}
=
e^{x - m_{\text{prev}}} \cdot e^{m_{\text{prev}} - m_{\text{new}}}
$$

定义：

$$
\alpha = e^{m_{\text{prev}} - m_{\text{new}}}
$$

那么旧分母在新参考系下就是：

$$
\ell_{\text{prev,new basis}} = \ell_{\text{prev}} \cdot \alpha
$$

对应代码：

```cpp
float oldScale = (runningSum == 0.0f) ? 0.0f : expf(runningMax - newMax);
```

---

## 10. runningSum 更新公式为什么成立

当前 tile 的贡献是：

$$
\ell_{\text{tile}} = \sum_{\text{tile } j} e^{s_{ij} - m_{\text{new}}}
$$

于是新的总分母是：

$$
\ell_{\text{new}}
=
\ell_{\text{prev}} \cdot \alpha + \ell_{\text{tile}}
$$

对应代码：

```cpp
float tileExpSum = 0.0f;
for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
    localScores[tileCol] = expf(localScores[tileCol] - newMax);
    tileExpSum += localScores[tileCol];
}

runningSum = runningSum * oldScale + tileExpSum;
runningMax = newMax;
```

这里注意：

- `localScores[tileCol]` 原本存的是 raw score
- 这一轮之后，它被原地改写成：

$$
e^{s_{ij} - m_{\text{new}}}
$$

这能减少后续重复计算。

---

## 11. 输出累积为什么也要 rescale

这是当前实现里最容易卡住的一步。

我们维护的输出分子是：

$$
a = \sum_j e^{s_{ij} - m} v_j
$$

假设旧状态是：

$$
a_{\text{prev}} = \sum_{\text{old } j} e^{s_{ij} - m_{\text{prev}}} v_j
$$

当最大值基准从 $m_{\text{prev}}$ 变成 $m_{\text{new}}$ 时，旧分子也必须重缩放：

$$
a_{\text{prev,new basis}}
=
a_{\text{prev}} \cdot e^{m_{\text{prev}} - m_{\text{new}}}
=
a_{\text{prev}} \cdot \alpha
$$

而当前 tile 的贡献是：

$$
a_{\text{tile}}
=
\sum_{\text{tile } j} e^{s_{ij} - m_{\text{new}}} v_j
$$

于是新的输出分子就是：

$$
a_{\text{new}}
=
a_{\text{prev}} \cdot \alpha + a_{\text{tile}}
$$

这就是代码里这一句的来源：

```cpp
out[queryIdx * valueDim + dim] =
    out[queryIdx * valueDim + dim] * oldScale + tileWeightedSum;
```

其中：

$$
\text{tileWeightedSum}
=
\sum_{\text{tile } j} e^{s_{ij} - m_{\text{new}}} v_j[\text{dim}]
$$

---

## 12. 为什么最后还要除一次 runningSum

到所有 tile 都处理完后，我们已经维护好了：

$$
a = \sum_j e^{s_{ij} - m} v_j
$$

$$
\ell = \sum_j e^{s_{ij} - m}
$$

于是最终输出就是：

$$
o_i = \frac{a}{\ell}
$$

对应代码：

```cpp
if (runningSum > 0.0f) {
    for (int dim = 0; dim < valueDim; ++dim) {
        out[queryIdx * valueDim + dim] /= runningSum;
    }
}
```

这样就恢复出了真正的 softmax attention 输出。

---

## 13. 这版实现和标准 attention 的等价关系

标准 attention 是：

$$
o_i = \sum_j \text{softmax}(s_{ij}) v_j
$$

当前这版 blockwise / online-softmax 实现实际上是在做：

1. 不直接显式构造完整一行 $s_{ij}$
2. 分 tile 计算局部 score
3. 用 running max / running sum 把各 tile 拼起来
4. 同步维护输出分子
5. 最后统一归一化

所以它和标准定义在数学上是等价的，只是计算顺序和中间状态的组织方式不同。

---

## 14. 当前实现真正完成了什么

就数学上来说，当前 `blockwise_cross_attention.cu` 已经完成了 FlashAttention 思想里最关键的一部分：

- K/V tile 化
- online softmax
- 输出分子随参考系变化而 rescale

也就是说，虽然它还不是工业级高性能 FlashAttention kernel，但在算法结构上，已经不只是“像 attention”，而是真正走到了 FlashAttention 核心数学思路上。

---

## 15. 当前实现还没有做好的地方

从工程性能角度，它还有明显优化空间：

1. `Q` 仍然在内层循环中反复从 global memory 读取  
   更合理的做法通常是先缓存到寄存器。

2. 输出分子 `out[...]` 当前直接作为全局累积缓冲区  
   更高性能实现通常会先放寄存器，最后再写回 global memory。

3. 当前 block / tile 形状仍然比较教学化  
   例如 `queryTileRows`、`keyTileCols` 还需要继续 sweep。

4. `localScores` 目前放在 shared memory 中  
   这虽然方便工程化，但不一定是最优选择。

---

## 16. 代码中的变量和数学符号对照

### 当前 query row

- 数学：$q_i$
- 代码：`queryIdx`

### 当前 tile 内的局部 score

- 数学：$s_{i,t,j}$
- 代码：`localScores[tileCol]`

### 当前 tile 最大值

- 数学：$m_{\text{tile}}$
- 代码：`tileMax`

### 旧全局最大值

- 数学：$m_{\text{prev}}$
- 代码：`runningMax`

### 新全局最大值

- 数学：$m_{\text{new}}$
- 代码：`newMax`

### 缩放因子

- 数学：$\alpha = e^{m_{\text{prev}} - m_{\text{new}}}$
- 代码：`oldScale`

### 分母

- 数学：$\ell$
- 代码：`runningSum`

### 分子

- 数学：$a$
- 代码：`out[queryIdx * valueDim + dim]`（当前实现里把它当累积缓冲区使用）

---

## 17. 一句话总结

这版实现的数学本质是：

> 把 attention 的 softmax 分母和输出分子都改写成“相对于当前 running max 的表示”，然后按 tile 增量更新，最后再统一归一化恢复标准 attention 输出。

这正是 FlashAttention 最核心的数学支点。
