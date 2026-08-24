"""
Unit Tests and Gradient Checks for Grouped-Query Attention (GQA) and MQA
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layers import GroupedQueryAttention, repeat_kv, unrepeat_kv_grad
from tests.test_gradcheck import compute_numerical_gradient, relative_error


class TestGroupedQueryAttention(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_repeat_kv_and_unrepeat(self):
        B, H_kv, S, D = 2, 2, 4, 8
        n_rep = 3
        x = np.random.randn(B, H_kv, S, D).astype(np.float64)
        rep = repeat_kv(x, n_rep)
        self.assertEqual(rep.shape, (B, H_kv * n_rep, S, D))

        grad_out = np.random.randn(*rep.shape).astype(np.float64)
        grad_analytical = unrepeat_kv_grad(grad_out, n_rep)

        def loss_fn():
            return np.sum(repeat_kv(x, n_rep) * grad_out)

        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-5, f"repeat_kv grad check failed with error: {err}")

    def test_gqa_forward_shapes(self):
        # 1. Multi-Query Attention (H_kv = 1)
        gqa_mqa = GroupedQueryAttention(d_model=32, num_heads=4, num_kv_heads=1)
        x = np.random.randn(2, 6, 32).astype(np.float32)
        out_mqa = gqa_mqa.forward(x)
        self.assertEqual(out_mqa.shape, (2, 6, 32))

        # 2. Grouped-Query Attention (H_kv = 2, H = 8)
        gqa_grouped = GroupedQueryAttention(d_model=64, num_heads=8, num_kv_heads=2)
        x2 = np.random.randn(2, 8, 64).astype(np.float32)
        out_gqa = gqa_grouped.forward(x2)
        self.assertEqual(out_gqa.shape, (2, 8, 64))

    def test_gqa_gradcheck_input(self):
        gqa = GroupedQueryAttention(d_model=16, num_heads=4, num_kv_heads=2)
        x = np.random.randn(2, 4, 16).astype(np.float64)
        out = gqa.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float64)

        grad_analytical = gqa.backward(grad_out)

        def loss_fn():
            return np.sum(gqa.forward(x) * grad_out)

        grad_numerical = compute_numerical_gradient(loss_fn, x, eps=1e-5)
        err = relative_error(grad_analytical, grad_numerical)
        self.assertLess(err, 1e-4, f"GQA input grad check failed with error: {err}")


if __name__ == "__main__":
    unittest.main()
