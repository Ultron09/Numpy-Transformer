# 🧠 Technical Specification: YaRN Context Window Extension

## 📌 Executive Summary
YaRN (Yet another RoPE extensioN) scales rotary positional frequencies non-uniformly with temperature correction to prevent attention entropy dilution at extended contexts.

---

## 🔬 Mathematical Formulation & Algorithmic Design

$$\theta'_i = (1 - \gamma) \theta_i + \gamma \left( \frac{\theta_i}{s} \right)$$

### Key Algorithmic Invariants:
1. **Computational Complexity:** $O(N \cdot d)$ per step.
2. **Memory Footprint:** Minimizes intermediate allocations.
3. **Numerical Precision:** Formulated for zero-overflow stability.

---

## ⚡ Pure NumPy Implementation

```python
import numpy as np

def compute_yarn_frequencies(dim: int, base: float = 10000.0, scale: float = 4.0, alpha: float = 1.0, beta: float = 32.0) -> np.ndarray:
    """
    Computes YaRN interpolated RoPE inverse frequencies with non-uniform frequency band ramp.
    """
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    wavelengths = 2.0 * np.pi / inv_freq
    low_freq = wavelengths < (beta * scale)
    high_freq = wavelengths > (alpha * scale)
    ramp = np.clip((wavelengths - alpha * scale) / ((beta - alpha) * scale), 0.0, 1.0)
    adjusted_inv_freq = (1.0 - ramp) * inv_freq + ramp * (inv_freq / scale)
    return adjusted_inv_freq
```

---

*AirBorne Engineering Excellence • Autonomous Intelligence Core*
