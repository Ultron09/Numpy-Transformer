"""
Unit tests for SparseMoEFFN and NoisyTopKGating in moe.py
"""

import unittest
import numpy as np
from moe import SparseMoEFFN, NoisyTopKGating, ExpertFFN


class TestMoE(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.d_model = 16
        self.d_ff = 32
        self.num_experts = 4
        self.top_k = 2

    def test_expert_ffn_forward_backward(self):
        expert = ExpertFFN(self.d_model, self.d_ff)
        x = np.random.randn(5, self.d_model).astype(np.float32)
        out = expert.forward(x)
        self.assertEqual(out.shape, (5, self.d_model))
        
        grad_out = np.random.randn(5, self.d_model).astype(np.float32)
        grad_x = expert.backward(grad_out)
        self.assertEqual(grad_x.shape, (5, self.d_model))
        self.assertIsNotNone(expert.grad_W1)
        self.assertIsNotNone(expert.grad_W2)

    def test_noisy_top_k_gating(self):
        router = NoisyTopKGating(self.d_model, self.num_experts, top_k=self.top_k)
        x = np.random.randn(10, self.d_model).astype(np.float32)
        
        weights, indices, full_probs, aux_loss = router.forward(x, is_training=False)
        self.assertEqual(weights.shape, (10, self.top_k))
        self.assertEqual(indices.shape, (10, self.top_k))
        self.assertEqual(full_probs.shape, (10, self.num_experts))
        
        # Softmax weights along top-k should sum to 1.0 for each token
        np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones(10), rtol=1e-5)
        # Aux loss must be non-negative
        self.assertTrue(aux_loss >= 0.0)

    def test_sparse_moe_ffn_forward(self):
        moe = SparseMoEFFN(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )
        x = np.random.randn(2, 6, self.d_model).astype(np.float32)
        out = moe.forward(x, is_training=False)
        self.assertEqual(out.shape, (2, 6, self.d_model))
        self.assertTrue(moe.last_aux_loss >= 0.0)


if __name__ == '__main__':
    unittest.main()
