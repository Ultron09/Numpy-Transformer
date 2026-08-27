"""
Unit tests for ContrastiveDecoder in contrastive_decoding.py
"""

import unittest
import numpy as np
from contrastive_decoding import ContrastiveDecoder, ContrastiveDecodingStats


class MockLM:
    """Mock language model with configurable logits."""
    def __init__(self, vocab_size: int = 10, shift: int = 1):
        self.vocab_size = vocab_size
        self.shift = shift
        self.seq_length = 64

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        logits = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        for t in range(seq_len):
            last_tok = input_ids[0, t]
            target_idx = (last_tok + self.shift) % self.vocab_size
            logits[0, t, target_idx] = 10.0
        return logits


class TestContrastiveDecoding(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.vocab_size = 10
        self.expert_model = MockLM(vocab_size=self.vocab_size, shift=1)
        self.amateur_model = MockLM(vocab_size=self.vocab_size, shift=2)
        self.decoder = ContrastiveDecoder(alpha=0.5, beta=0.1, do_sample=False)

    def test_contrastive_scores_apc_masking(self):
        # Expert logits strongly favor token 0 and 1, token 5 has very low logit
        expert_logits = np.array([10.0, 9.5, 2.0, 1.0, 0.0, -10.0], dtype=np.float32)
        amateur_logits = np.array([5.0, 1.0, 8.0, 3.0, 0.0, 2.0], dtype=np.float32)

        scores, mask = self.decoder.compute_contrastive_scores(expert_logits, amateur_logits)

        # Tokens with very low probability under expert should be masked (-inf)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[5])
        self.assertEqual(scores[5], -np.inf)

    def test_amateur_suppression(self):
        # Expert gives token A=0 and token B=1 roughly equal high logits (5.0 vs 4.8)
        # Amateur strongly predicts token A=0 (10.0) but low on B=1 (1.0)
        # Contrastive decoding should penalize token A and select token B
        expert_logits = np.array([5.0, 4.8, -2.0, -2.0], dtype=np.float32)
        amateur_logits = np.array([10.0, 1.0, -2.0, -2.0], dtype=np.float32)

        decoder = ContrastiveDecoder(alpha=1.0, beta=0.1, do_sample=False)
        selected_token, _ = decoder.select_next_token(expert_logits, amateur_logits)

        self.assertEqual(selected_token, 1)

    def test_sampling_mode(self):
        expert_logits = np.array([5.0, 4.8, 4.5, 0.0], dtype=np.float32)
        amateur_logits = np.array([2.0, 2.0, 2.0, 0.0], dtype=np.float32)

        decoder = ContrastiveDecoder(alpha=0.5, beta=0.1, temperature=1.0, do_sample=True)
        selected_token, num_plausible = decoder.select_next_token(expert_logits, amateur_logits)

        self.assertIn(selected_token, [0, 1, 2])
        self.assertEqual(num_plausible, 3)

    def test_end_to_end_generation(self):
        prompt = [1]
        tokens, stats = self.decoder.generate(
            expert_model=self.expert_model,
            amateur_model=self.amateur_model,
            prompt_ids=prompt,
            max_new_tokens=5,
        )

        self.assertEqual(len(tokens), 6)
        self.assertEqual(stats.total_tokens_generated, 5)
        self.assertEqual(stats.num_steps, 5)
        self.assertTrue(stats.avg_plausibility_set_size >= 1.0)

    def test_eos_token_stopping(self):
        # Stop at token 3
        prompt = [1]
        tokens, stats = self.decoder.generate(
            expert_model=self.expert_model,
            amateur_model=self.amateur_model,
            prompt_ids=prompt,
            max_new_tokens=10,
            eos_token_id=3,
        )

        self.assertEqual(tokens[-1], 3)
        self.assertTrue(len(tokens) < 11)


if __name__ == '__main__':
    unittest.main()
