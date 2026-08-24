"""
Unit Tests for Advanced Attention Mechanisms (Sliding Window, Tiled Online Softmax)
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attention_mechanisms import sliding_window_causal_mask, tiled_online_softmax_attention, SlidingWindowAttention
from kv_cache import scaled_dot_product_attention


class TestAttentionMechanisms(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_sliding_window_mask_bounds(self):
        seq_len = 6
        window_size = 3
        mask = sliding_window_causal_mask(seq_len, window_size)[0, 0]
        
        # Position 0 can only attend to position 0
        self.assertEqual(mask[0, 0], 0.0)
        self.assertEqual(mask[0, 1], -1e9)
        
        # Position 4 can attend to positions 2, 3, 4 (window of 3)
        self.assertEqual(mask[4, 0], -1e9)
        self.assertEqual(mask[4, 1], -1e9)
        self.assertEqual(mask[4, 2], 0.0)
        self.assertEqual(mask[4, 3], 0.0)
        self.assertEqual(mask[4, 4], 0.0)
        self.assertEqual(mask[4, 5], -1e9)

    def test_tiled_online_softmax_numerical_equivalence(self):
        B, H, S, D = 2, 4, 32, 16
        q = np.random.randn(B, H, S, D).astype(np.float32)
        k = np.random.randn(B, H, S, D).astype(np.float32)
        v = np.random.randn(B, H, S, D).astype(np.float32)

        standard_out, _ = scaled_dot_product_attention(q, k, v)
        tiled_out = tiled_online_softmax_attention(q, k, v, block_size_q=8, block_size_kv=8)

        max_diff = np.max(np.abs(standard_out - tiled_out))
        self.assertLess(max_diff, 1e-5, f"Tiled online softmax differs from standard attention: {max_diff}")

    def test_sliding_window_attention_forward(self):
        swa = SlidingWindowAttention(d_model=32, num_heads=4, window_size=8)
        x = np.random.randn(2, 16, 32).astype(np.float32)
        out = swa.forward(x)
        self.assertEqual(out.shape, (2, 16, 32))


if __name__ == "__main__":
    unittest.main()
