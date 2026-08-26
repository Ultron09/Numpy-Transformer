"""
Speculative Decoding Engine for Accelerating Autoregressive Transformer Inference

Implements Speculative Sampling (Leviathan et al., 2023; Chen et al., 2023) in pure NumPy.
Employs a lightweight 'draft' model to rapidly generate candidate token sequences,
then verifies them in parallel with a single forward pass of the larger 'target' model.

Mathematical Invariance:
    Speculative sampling provably guarantees that the output distribution is IDENTICAL
    to sampling directly from the target model:
    
    1. Acceptance probability:
        α = min(1, p(x) / q(x))
        where p(x) is target distribution, q(x) is draft distribution.
        
    2. Rejection residual distribution:
        p'(x) = norm(max(0, p(x) - q(x)))
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class SpeculativeDecodingStats:
    """Statistics tracking performance of speculative decoding."""
    total_tokens_generated: int = 0
    total_draft_tokens_proposed: int = 0
    total_draft_tokens_accepted: int = 0
    num_target_forward_passes: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens_proposed == 0:
            return 0.0
        return self.total_draft_tokens_accepted / self.total_draft_tokens_proposed
        
    @property
    def tokens_per_step(self) -> float:
        if self.num_target_forward_passes == 0:
            return 0.0
        return self.total_tokens_generated / self.num_target_forward_passes


class SpeculativeDecoder:
    """
    Pure NumPy Speculative Decoding verification engine.
    """
    
    def __init__(
        self,
        gamma: int = 4,
        temperature: float = 1.0,
    ):
        """
        Args:
            gamma: Number of draft tokens speculated per iteration (lookahead window)
            temperature: Sampling temperature for both models
        """
        self.gamma = gamma
        self.temperature = max(temperature, 1e-5)
        
    def _get_probs(self, logits: np.ndarray) -> np.ndarray:
        """Compute stable softmax probability distribution from logits."""
        scaled = logits / self.temperature
        max_val = np.max(scaled, axis=-1, keepdims=True)
        exp_vals = np.exp(scaled - max_val)
        probs = exp_vals / (np.sum(exp_vals, axis=-1, keepdims=True) + 1e-12)
        return probs

    def _sample_from_probs(self, probs: np.ndarray) -> int:
        """Categorical sample from probability array."""
        probs = np.nan_to_num(probs, nan=0.0)
        p_sum = np.sum(probs)
        if p_sum <= 0:
            return int(np.argmax(probs))
        probs = probs / p_sum
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        draft_model,
        target_model,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], SpeculativeDecodingStats]:
        """
        Execute speculative decoding.
        
        Args:
            draft_model: Fast compact model for drafting tokens
            target_model: Accurate primary model for parallel verification
            prompt_ids: Input prompt token IDs
            max_new_tokens: Maximum target tokens to generate
            eos_token_id: Optional End-of-Sequence token ID
            
        Returns:
            generated_ids: Full list of generated token IDs
            stats: SpeculativeDecodingStats performance metrics
        """
        stats = SpeculativeDecodingStats()
        generated = list(prompt_ids)
        target_len = len(prompt_ids) + max_new_tokens
        
        while len(generated) < target_len:
            curr_prompt_len = len(generated)
            # Step 1: Draft model generates γ candidate tokens autoregressively
            draft_tokens = []
            draft_probs_list = []
            
            curr_draft_seq = list(generated)
            for _ in range(self.gamma):
                if len(curr_draft_seq) >= target_len:
                    break
                # Forward draft model
                draft_in = np.array([curr_draft_seq], dtype=np.int32)
                draft_logits = draft_model.forward(draft_in)[0, -1, :]
                draft_p = self._get_probs(draft_logits)
                
                draft_tok = self._sample_from_probs(draft_p)
                draft_tokens.append(draft_tok)
                draft_probs_list.append(draft_p)
                curr_draft_seq.append(draft_tok)
                
                if eos_token_id is not None and draft_tok == eos_token_id:
                    break
                    
            if not draft_tokens:
                break
                
            stats.total_draft_tokens_proposed += len(draft_tokens)
            
            # Step 2: Target model runs 1 parallel forward pass over the speculated sequence
            target_in = np.array([curr_draft_seq], dtype=np.int32)
            target_logits_all = target_model.forward(target_in)[0]  # shape (seq_len, vocab_size)
            stats.num_target_forward_passes += 1
            
            # Slice the target logits corresponding to verification positions
            # Position (curr_prompt_len - 1) gives prediction for draft_tokens[0]
            # Position (curr_prompt_len + k - 1) gives prediction for draft_tokens[k]
            all_accepted = True
            
            for k, (tok, q_p) in enumerate(zip(draft_tokens, draft_probs_list)):
                target_pos = curr_prompt_len - 1 + k
                target_logits = target_logits_all[target_pos]
                target_p = self._get_probs(target_logits)
                
                p_x = target_p[tok]
                q_x = q_p[tok]
                
                # Acceptance ratio: α = min(1, p(x) / q(x))
                alpha = min(1.0, float(p_x / (q_x + 1e-12)))
                r = np.random.uniform(0.0, 1.0)
                
                if r < alpha:
                    # Accept token
                    generated.append(tok)
                    stats.total_draft_tokens_accepted += 1
                    stats.total_tokens_generated += 1
                    if eos_token_id is not None and tok == eos_token_id:
                        all_accepted = False
                        break
                else:
                    # Reject token: resample from adjusted distribution: p'(x) = norm(relu(p(x) - q(x)))
                    all_accepted = False
                    diff = np.maximum(0.0, target_p - q_p)
                    diff_sum = np.sum(diff)
                    if diff_sum > 0:
                        resampled_p = diff / diff_sum
                    else:
                        resampled_p = target_p
                        
                    resampled_tok = self._sample_from_probs(resampled_p)
                    generated.append(resampled_tok)
                    stats.total_tokens_generated += 1
                    break
                    
            # If all γ draft tokens were accepted, sample 1 bonus token from final target position
            if all_accepted and len(generated) < target_len:
                final_pos = curr_prompt_len - 1 + len(draft_tokens)
                if final_pos < len(target_logits_all):
                    final_target_p = self._get_probs(target_logits_all[final_pos])
                    bonus_tok = self._sample_from_probs(final_target_p)
                    generated.append(bonus_tok)
                    stats.total_tokens_generated += 1
                    
            if eos_token_id is not None and generated[-1] == eos_token_id:
                break
                
        return generated, stats
