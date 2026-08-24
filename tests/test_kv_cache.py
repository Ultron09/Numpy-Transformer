"""
Unit Tests for KV Cache and Cached Autoregressive Generation
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kv_cache import KVCache, LayerKVCacheManager, CachedModernTransformer


class TestKVCache(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_kv_cache_update_and_reset(self):
        cache = KVCache(max_batch_size=2, max_seq_len=16, num_heads=4, head_dim=8)
        k1 = np.random.randn(2, 4, 3, 8).astype(np.float32)
        v1 = np.random.randn(2, 4, 3, 8).astype(np.float32)
        
        all_k, all_v = cache.update(k1, v1)
        self.assertEqual(all_k.shape, (2, 4, 3, 8))
        self.assertEqual(cache.current_len, 3)

        k2 = np.random.randn(2, 4, 1, 8).astype(np.float32)
        v2 = np.random.randn(2, 4, 1, 8).astype(np.float32)
        all_k, all_v = cache.update(k2, v2)
        self.assertEqual(all_k.shape, (2, 4, 4, 8))
        self.assertEqual(cache.current_len, 4)

        cache.reset()
        self.assertEqual(cache.current_len, 0)

    def test_kv_cache_overflow_error(self):
        cache = KVCache(max_batch_size=1, max_seq_len=4, num_heads=2, head_dim=4)
        k = np.random.randn(1, 2, 5, 4).astype(np.float32)
        v = np.random.randn(1, 2, 5, 4).astype(np.float32)
        with self.assertRaises(ValueError):
            cache.update(k, v)

    def test_cached_generation_logits_equivalence(self):
        model = CachedModernTransformer(
            vocab_size=30,
            d_model=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64
        )
        prompt = [2, 7, 14]
        
        # 1. Full non-cached forward pass
        full_logits = model.forward(np.array([prompt]))

        # 2. Step-by-step KV-cache forward pass
        cache_mgr = model.create_cache_manager(batch_size=1)
        step_logits = None
        for i, tok in enumerate(prompt):
            step_logits = model.forward_step(np.array([[tok]]), cache_mgr, pos_offset=i)

        max_abs_diff = np.max(np.abs(full_logits[0, -1, :] - step_logits[0, -1, :]))
        self.assertLess(max_abs_diff, 1e-4, f"Cached logits diverged: {max_abs_diff}")

    def test_cached_text_generation(self):
        model = CachedModernTransformer(
            vocab_size=20,
            d_model=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=32
        )
        prompt = [1, 3]
        generated = model.generate_cached(prompt, max_new_tokens=6)
        self.assertEqual(len(generated), len(prompt) + 6)
        self.assertEqual(generated[:2], prompt)


if __name__ == "__main__":
    unittest.main()
