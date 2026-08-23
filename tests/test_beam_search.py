"""
Unit tests for BeamSearchDecoder and BeamHypothesis in sampler.py
"""

import unittest
import numpy as np
from sampler import BeamHypothesis, BeamSearchDecoder


class DummyModel:
    """Mock model with deterministic token distributions for testing beam search."""
    def __init__(self, vocab_size: int = 10, seq_length: int = 32):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        # input_ids: shape (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape
        last_token = input_ids[0, -1]
        
        # Build predictable logits based on current token sequence
        logits = np.zeros((batch_size, seq_len, self.vocab_size), dtype=np.float32)
        # Give highest probability to (last_token + 1) % vocab_size, second to (last_token + 2)
        logits[0, -1, (last_token + 1) % self.vocab_size] = 5.0
        logits[0, -1, (last_token + 2) % self.vocab_size] = 3.0
        logits[0, -1, 0] = 1.0  # token 0 as potential EOS
        return logits


class TestBeamSearch(unittest.TestCase):
    
    def test_beam_hypothesis_scoring(self):
        hyp = BeamHypothesis(tokens=[1, 2, 3, 4], log_prob=-2.4)
        self.assertEqual(hyp.length, 4)
        # Length penalty 0
        self.assertAlmostEqual(hyp.compute_score(length_penalty=0.0, prompt_len=1), -2.4)
        # Length penalty > 0
        score = hyp.compute_score(length_penalty=1.0, prompt_len=1)
        self.assertTrue(score > -2.4)

    def test_beam_search_decoding(self):
        model = DummyModel(vocab_size=10)
        decoder = BeamSearchDecoder(beam_width=3, max_new_tokens=4, length_penalty=1.0)
        results = decoder.search(model, prompt_ids=[1], num_return_sequences=2)
        
        self.assertEqual(len(results), 2)
        best_tokens, best_score = results[0]
        self.assertTrue(len(best_tokens) > 1)
        self.assertEqual(best_tokens[0], 1)
        # Best sequence should follow highest logits: 1 -> 2 -> 3 -> 4 -> 5
        self.assertEqual(best_tokens, [1, 2, 3, 4, 5])

    def test_ngram_blocking(self):
        model = DummyModel(vocab_size=10)
        decoder = BeamSearchDecoder(beam_width=2, max_new_tokens=6, no_repeat_ngram_size=2)
        results = decoder.search(model, prompt_ids=[1, 2, 3], num_return_sequences=1)
        best_tokens, _ = results[0]
        # Check that no 2-gram repeats in the generated sequence
        ngrams = [tuple(best_tokens[i:i+2]) for i in range(len(best_tokens)-1)]
        self.assertEqual(len(ngrams), len(set(ngrams)))

    def test_eos_termination(self):
        model = DummyModel(vocab_size=10)
        decoder = BeamSearchDecoder(beam_width=2, max_new_tokens=10, early_stopping=True)
        # Set EOS token id = 3
        results = decoder.search(model, prompt_ids=[1], eos_token_id=3, num_return_sequences=1)
        best_tokens, _ = results[0]
        self.assertIn(3, best_tokens)
        self.assertEqual(best_tokens[-1], 3)


if __name__ == '__main__':
    unittest.main()
