"""
Numerical Gradient Checking Suite

Verifies analytical backward passes against centered finite difference approximations:
    df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
"""

import sys
import os
import unittest
import numpy as np

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt_numpy import GELU, Softmax, LayerNorm, Linear
from train import cross_entropy_loss
from layers import RMSNorm, SiLU, SwiGLU
from positional_embeddings import RotaryEmbedding


def compute_numerical_gradient(forward_fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient using centered finite differences."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]
        
        x[idx] = orig_val + eps
        fx_plus = forward_fn()
        
        x[idx] = orig_val - eps
        fx_minus = forward_fn()
        
        x[idx] = orig_val  # restore
        
        grad[idx] = (fx_plus - fx_minus) / (2.0 * eps)
        it.iternext()
        
    return grad


def relative_error(grad_analytical: np.ndarray, grad_numerical: np.ndarray) -> float:
    """Compute relative error between analytical and numerical gradients."""
    numerator = np.linalg.norm(grad_analytical - grad_numerical)
    denominator = np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical) + 1e-10
    return float(numerator / denominator)


class TestGradients(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_gelu_grad(self):
        gelu = GELU()
        x = np.random.randn(3, 4).astype(np.float64)
        out = gelu.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = gelu.backward(grad_out)
        
        def loss_fn():
            return np.sum(gelu.forward(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-5, f"GELU grad check failed with relative error: {err}")

    def test_silu_grad(self):
        silu = SiLU()
        x = np.random.randn(3, 5).astype(np.float64)
        out = silu.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = silu.backward(grad_out)
        
        def loss_fn():
            return np.sum(silu.forward(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-5, f"SiLU grad check failed with relative error: {err}")

    def test_rmsnorm_grad(self):
        norm = RMSNorm(d_model=6)
        x = np.random.randn(2, 4, 6).astype(np.float64)
        out = norm.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = norm.backward(grad_out)
        
        def loss_fn():
            return np.sum(norm.forward(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-4, f"RMSNorm grad check failed with relative error: {err}")

    def test_linear_grad(self):
        layer = Linear(in_features=5, out_features=4, bias=True)
        x = np.random.randn(2, 3, 5).astype(np.float64)
        out = layer.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = layer.backward(grad_out)
        
        def loss_fn():
            return np.sum(layer.forward(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-5, f"Linear grad check failed with relative error: {err}")

    def test_cross_entropy_grad(self):
        logits = np.random.randn(2, 3, 5).astype(np.float64)
        targets = np.array([[0, 3, 2], [1, 4, 0]], dtype=np.int64)
        
        loss, grad_analytical = cross_entropy_loss(logits, targets)
        
        def loss_fn():
            l, _ = cross_entropy_loss(logits, targets)
            return l
            
    def test_layernorm_grad(self):
        ln = LayerNorm(normalized_shape=6)
        x = np.random.randn(2, 3, 6).astype(np.float64)
        out = ln.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = ln.backward(grad_out)
        
        def loss_fn():
            return np.sum(ln.forward(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-4, f"LayerNorm grad check failed with relative error: {err}")

    def test_swiglu_grad(self):
        glu = SwiGLU(d_model=6, hidden_dim=8)
        x = np.random.randn(2, 3, 6).astype(np.float64)
        out = glu.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = glu.backward(grad_out)
        
        def loss_fn():
            return np.sum(glu.forward(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-4, f"SwiGLU grad check failed with relative error: {err}")

    def test_rope_grad(self):
        rope = RotaryEmbedding(dim=6)
        x = np.random.randn(1, 2, 4, 6).astype(np.float64)
        out = rope.apply_rope(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)
        
        grad_analytical = rope.backward_rope(grad_out)
        
        def loss_fn():
            return np.sum(rope.apply_rope(x) * grad_out)
            
        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-5, f"RoPE grad check failed with relative error: {err}")


if __name__ == "__main__":
    unittest.main()
