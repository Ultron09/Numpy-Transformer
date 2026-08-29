# 🧠 Technical Specification: ALiBi Attention with Linear Biases Geometric Decay

## 📌 Executive Summary
ALiBi enforces position recency by subtracting a static, query-key distance penalty weighted by head-specific geometric decay slopes.

---

## 🔬 Mathematical Formulation & Algorithmic Design

$$A_{i, j} = q_i k_j^T - m \cdot |i - j|, \quad m = 2^{-\frac{8 h}{H}}$$

### Key Algorithmic Invariants:
1. **Computational Complexity:** $O(N \cdot d)$ per step.
2. **Memory Footprint:** Minimizes intermediate allocations.
3. **Numerical Precision:** Formulated for zero-overflow stability.

---

## ⚡ Pure NumPy Implementation

```python
import numpy as np

def get_alibi_slopes(num_heads: int) -> list:
    ratio = 2.0 ** (-8.0 / num_heads)
    return [ratio ** (i + 1) for i in range(num_heads)]
```

---

*AirBorne Engineering Excellence • Autonomous Intelligence Core*
