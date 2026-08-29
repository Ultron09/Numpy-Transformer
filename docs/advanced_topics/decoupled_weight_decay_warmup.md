# 🧠 Technical Specification: Decoupled Weight Decay with Cosine Warmup Dynamics

## 📌 Executive Summary
AdamW decouples weight regularization from stochastic gradient variance, paired with cosine learning rate warmup schedules for optimal generalization.

---

## 🔬 Mathematical Formulation & Algorithmic Design

$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

### Key Algorithmic Invariants:
1. **Computational Complexity:** $O(N \cdot d)$ per step.
2. **Memory Footprint:** Minimizes intermediate allocations.
3. **Numerical Precision:** Formulated for zero-overflow stability.

---

## ⚡ Pure NumPy Implementation

```python
import numpy as np

def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return max_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
```

---

*AirBorne Engineering Excellence • Autonomous Intelligence Core*
