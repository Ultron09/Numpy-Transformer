"""
Unit tests for LoRALinear in lora.py
"""

import unittest
import numpy as np
from lora import LoRALinear


class TestLoRALinear(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.in_dim = 16
        self.out_dim = 24
        self.rank = 4
        self.alpha = 8.0
        self.layer = LoRALinear(self.in_dim, self.out_dim, r=self.rank, lora_alpha=self.alpha)

    def test_initialization_zero_delta(self):
        # Since lora_B is initialized to 0, LoRA output must match base W output at start
        x = np.random.randn(2, 5, self.in_dim).astype(np.float32)
        out = self.layer.forward(x)
        expected_base = np.matmul(x, self.layer.W) + self.layer.bias
        np.testing.assert_allclose(out, expected_base, rtol=1e-5, atol=1e-5)

    def test_forward_with_active_lora(self):
        # Set non-zero lora_B
        self.layer.lora_B = np.random.randn(self.rank, self.out_dim).astype(np.float32) * 0.1
        x = np.random.randn(3, self.in_dim).astype(np.float32)
        out = self.layer.forward(x)
        
        # Calculate expected output
        delta_W = (self.alpha / self.rank) * np.matmul(self.layer.lora_A, self.layer.lora_B)
        expected = np.matmul(x, self.layer.W + delta_W) + self.layer.bias
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_merge_unmerge_invariance(self):
        self.layer.lora_B = np.random.randn(self.rank, self.out_dim).astype(np.float32) * 0.1
        x = np.random.randn(2, 4, self.in_dim).astype(np.float32)
        
        out_unmerged = self.layer.forward(x)
        self.layer.merge_weights()
        self.assertTrue(self.layer.merged)
        
        out_merged = self.layer.forward(x)
        np.testing.assert_allclose(out_unmerged, out_merged, rtol=1e-5, atol=1e-5)
        
        self.layer.unmerge_weights()
        self.assertFalse(self.layer.merged)
        out_unmerged_again = self.layer.forward(x)
        np.testing.assert_allclose(out_unmerged, out_unmerged_again, rtol=1e-5, atol=1e-5)

    def test_gradients_numerical(self):
        self.layer.lora_B = np.random.randn(self.rank, self.out_dim).astype(np.float32) * 0.1
        x = np.random.randn(2, self.in_dim).astype(np.float32)
        grad_out = np.random.randn(2, self.out_dim).astype(np.float32)
        
        # Forward and analytical backward
        self.layer.forward(x)
        grad_x = self.layer.backward(grad_out)
        
        # Check gradient w.r.t lora_A via finite differences
        eps = 1e-4
        i, j = 1, 2
        orig_val = self.layer.lora_A[i, j]
        
        self.layer.lora_A[i, j] = orig_val + eps
        out_pos = self.layer.forward(x)
        loss_pos = np.sum(out_pos * grad_out)
        
        self.layer.lora_A[i, j] = orig_val - eps
        out_neg = self.layer.forward(x)
        loss_neg = np.sum(out_neg * grad_out)
        
        self.layer.lora_A[i, j] = orig_val
        num_grad_A = (loss_pos - loss_neg) / (2 * eps)
        
        np.testing.assert_allclose(self.layer.grad_lora_A[i, j], num_grad_A, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
