# 🧠 Technical Specification: Finite State Machine Guided Decoding for Regex

## 📌 Executive Summary
FSM-guided decoding maps regular expressions to state machines, masking out vocabulary logits that would transition into invalid grammatical states.

---

## 🔬 Mathematical Formulation & Algorithmic Design

$$\mathcal{M}(s, v) = \begin{cases} 0 & \text{if } \delta(s, v) \neq \emptyset \\ -\infty & \text{otherwise} \end{cases}$$

### Key Algorithmic Invariants:
1. **Computational Complexity:** $O(N \cdot d)$ per step.
2. **Memory Footprint:** Minimizes intermediate allocations.
3. **Numerical Precision:** Formulated for zero-overflow stability.

---

## ⚡ Pure NumPy Implementation

```python
import numpy as np

def mask_logits_with_fsm(logits: np.ndarray, allowed_tokens: set) -> np.ndarray:
    mask = np.full_like(logits, -1e9)
    for tok in allowed_tokens:
        mask[tok] = 0.0
    return logits + mask
```

---

*AirBorne Engineering Excellence • Autonomous Intelligence Core*
