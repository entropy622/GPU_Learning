# Online Softmax And Output Accumulation Notes

This note explains why the following update rules are correct in a FlashAttention-style blockwise implementation.

## 1. Standard Softmax

For one query row, suppose the attention logits are:

$$
x_1, x_2, \dots, x_n
$$

Then softmax is:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

The attention output is:

$$
O = \sum_j \text{softmax}(x_j) V_j
$$

Expanding softmax:

$$
O = \frac{\sum_j e^{x_j} V_j}{\sum_j e^{x_j}}
$$

So the output is:

$$
O = \frac{\text{numerator}}{\text{denominator}}
$$

where:

$$
\text{numerator} = \sum_j e^{x_j} V_j
$$

and

$$
\text{denominator} = \sum_j e^{x_j}
$$

## 2. Why Introduce Max?

Softmax itself does **not** contain a max operation.

We introduce max for numerical stability.

Softmax has the shift-invariance property:

$$
\text{softmax}(x_i) = \text{softmax}(x_i - c)
$$

for any constant $c$ shared by the whole row.

Proof:

$$
\frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
=
\frac{e^{x_i} e^{-c}}{\sum_j e^{x_j} e^{-c}}
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

So we can choose:

$$
c = \max_j x_j
$$

and rewrite softmax as:

$$
\text{softmax}(x_i)
=
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
\quad\text{where}\quad
m = \max_j x_j
$$

This keeps the exponentials stable because now all terms satisfy:

$$
x_i - m \le 0
$$

## 3. Why Online Softmax Is Needed

In a blockwise algorithm, we do not process the whole row at once.

Instead, we scan the keys in tiles:

$$
\text{tile}_1, \text{tile}_2, \text{tile}_3, \dots
$$

So we do not know the final global max at the beginning.

To make blockwise processing possible, we maintain:

- a running max
- a running denominator

## 4. Meaning Of The Running State

After processing some tiles, define:

$$
m_{\text{prev}} = \text{runningMax}
$$

$$
\ell_{\text{prev}} = \text{runningSum}
$$

The meaning of $\ell_{\text{prev}}$ is:

$$
\ell_{\text{prev}} = \sum_{\text{old } j} e^{x_j - m_{\text{prev}}}
$$

So it is **not** the raw sum $\sum e^{x_j}$.

It is the sum of exponentials expressed relative to the current running max.

## 5. Current Tile Statistics

Suppose the current tile contains logits:

$$
x_{t,1}, x_{t,2}, \dots, x_{t,k}
$$

Its local max is:

$$
m_{\text{tile}} = \max(x_{t,1}, \dots, x_{t,k})
$$

The new global max after merging old tiles and current tile is:

$$
m_{\text{new}} = \max(m_{\text{prev}}, m_{\text{tile}})
$$

## 6. Why The Old Sum Must Be Rescaled

Previously, the old denominator was represented under the basis $m_{\text{prev}}$:

$$
\ell_{\text{prev}} = \sum_{\text{old } j} e^{x_j - m_{\text{prev}}}
$$

Now we want everything to be represented under the new basis $m_{\text{new}}$.

For any old element:

$$
e^{x_j - m_{\text{new}}}
=
e^{x_j - m_{\text{prev}}} \cdot e^{m_{\text{prev}} - m_{\text{new}}}
$$

Therefore:

$$
\sum_{\text{old } j} e^{x_j - m_{\text{new}}}
=
\left(\sum_{\text{old } j} e^{x_j - m_{\text{prev}}}\right)
\cdot e^{m_{\text{prev}} - m_{\text{new}}}
$$

So:

$$
\text{old contribution under new basis}
=
\ell_{\text{prev}} \cdot e^{m_{\text{prev}} - m_{\text{new}}}
$$

Define:

$$
\alpha = e^{m_{\text{prev}} - m_{\text{new}}}
$$

Then the denominator update becomes:

$$
\ell_{\text{new}} = \ell_{\text{prev}} \alpha + \sum_{\text{tile } j} e^{x_j - m_{\text{new}}}
$$

This is exactly the code pattern:

```cpp
float newMax = fmaxf(runningMax, tileMax);
float oldScale = expf(runningMax - newMax);
runningSum = runningSum * oldScale + tileExpSum;
runningMax = newMax;
```

## 7. Why Output Accumulation Needs The Same Rescaling

The attention output numerator is:

$$
\sum_j e^{x_j} V_j
$$

Under running-max normalization, we instead maintain:

$$
\text{acc}_{\text{prev}} = \sum_{\text{old } j} e^{x_j - m_{\text{prev}}} V_j
$$

This is a vector, not a scalar.

When the basis changes from $m_{\text{prev}}$ to $m_{\text{new}}$, the old numerator must be rescaled in exactly the same way:

$$
\sum_{\text{old } j} e^{x_j - m_{\text{new}}} V_j
=
\left(\sum_{\text{old } j} e^{x_j - m_{\text{prev}}} V_j\right)
\cdot e^{m_{\text{prev}} - m_{\text{new}}}
$$

So:

$$
\text{acc}_{\text{prev,new basis}} = \text{acc}_{\text{prev}} \cdot \alpha
$$

The current tile contribution is:

$$
\text{acc}_{\text{tile}} = \sum_{\text{tile } j} e^{x_j - m_{\text{new}}} V_j
$$

Therefore the output numerator update is:

$$
\text{acc}_{\text{new}} = \text{acc}_{\text{prev}} \cdot \alpha + \text{acc}_{\text{tile}}
$$

This is the meaning of:

$$
\boxed{\text{acc}_{\text{new}} = \text{acc}_{\text{prev}} \cdot \text{oldScale} + \text{currentTileContrib}}
$$

## 8. Final Output

After all tiles have been processed, we have:

$$
\ell = \sum_j e^{x_j - m}
$$

and

$$
\text{acc} = \sum_j e^{x_j - m} V_j
$$

So the final attention output is:

$$
O = \frac{\text{acc}}{\ell}
$$

This works because both numerator and denominator are represented under the same basis $m$.

## 9. Mapping To Code

The typical blockwise pattern is:

```cpp
float runningMax = -FLT_MAX;
float runningSum = 0.0f;
// out[...] temporarily stores the numerator accumulator acc

for (each K/V tile) {
    // Step 1: load keyTile and valueTile into shared memory

    // Step 2: compute tile-local scores
    // localScores[tileCol] = score
    // tileMax = max score in current tile

    float newMax = fmaxf(runningMax, tileMax);
    float oldScale = (runningSum == 0.0f) ? 0.0f : expf(runningMax - newMax);

    float tileExpSum = 0.0f;
    for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
        localScores[tileCol] = expf(localScores[tileCol] - newMax);
        tileExpSum += localScores[tileCol];
    }

    for (int dim = 0; dim < valueDim; ++dim) {
        float tileWeightedSum = 0.0f;
        for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
            tileWeightedSum += localScores[tileCol] * valueTile[tileCol * valueDim + dim];
        }

        out[queryIdx * valueDim + dim] =
            out[queryIdx * valueDim + dim] * oldScale + tileWeightedSum;
    }

    runningSum = runningSum * oldScale + tileExpSum;
    runningMax = newMax;
}

for (int dim = 0; dim < valueDim; ++dim) {
    out[queryIdx * valueDim + dim] /= runningSum;
}
```

## 10. One-Sentence Summary

The max does not come from the definition of softmax.

It is introduced so that softmax can be computed stably and incrementally, tile by tile, while keeping all partial sums and output accumulators in the same normalization basis.
