"""
Unit tests for SpeculativeDecoder in speculative_decoding.py
"""

import unittest
import numpy as np
from speculative_decoding import SpeculativeDecoder, SpeculativeDecodingStats


class MockLM:
    """Mock language model with configurable logits."""
    def __init__(self, vocab_size: int = 10, shift: int = 1):
        self.vocab_size = vocab_size
        self.shift = shift
        
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        logits = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        for t in range(seq_len):
            last_tok = input_ids[0, t]
            target_idx = (last_tok + self.shift) % self.vocab_size
            logits[0, t, target_idx] = 10.0  # highly peaked distribution
        return logits


class TestSpeculativeDecoding(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.vocab_size = 10
        self.draft_model = MockLM(vocab_size=self.vocab_size, shift=1)
        self.target_model = MockLM(vocab_size=self.vocab_size, shift=1)
        self.decoder = SpeculativeDecoder(gamma=3, temperature=0.1)

    def test_perfect_alignment_acceptance(self):
        # When draft and target predict identical distributions, acceptance rate should be ~100%
        prompt = [1]
        tokens, stats = self.decoder.generate(
            draft_model=self.draft_model,
            target_model=self.target_model,
            prompt_ids=prompt,
            max_new_tokens=6,
        )
        self.assertTrue(len(tokens) >= 7)
        self.assertAlmostEqual(stats.acceptance_rate, 1.0, places=2)
        self.assertTrue(stats.tokens_per_step > 1.0)

    def test_rejection_and_resampling(self):
        # Target model predicts different tokens than draft model
        divergent_target = MockLM(vocab_size=self.vocab_size, shift=2)
        prompt = [1]
        tokens, stats = self.decoder.generate(
            draft_model=self.draft_model,
            target_model=divergent_target,
            prompt_ids=prompt,
            max_new_tokens=5,
        )
        self.assertTrue(len(tokens) >= 6)
        # Should have lower acceptance rate due to model mismatch
        self.assertTrue(stats.num_target_forward_passes > 0)


if __name__ == '__main__':
    unittest.main()
