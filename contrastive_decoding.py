"""
Contrastive Decoding for Autoregressive Language Models

Implements Contrastive Decoding (Li et al., ACL 2023: "Contrastive Decoding: Open-ended Text Generation as Optimization").
Uses a smaller 'amateur' (draft) model to identify and penalize undesirable linguistic artifacts,
repetition loops, and degenerate tokens produced by the 'expert' model.

Mathematical Formulation:
1. Adaptive Plausibility Constraint (APC):
    V_head(x) = { v in V | P_exp(v | x) >= beta * max_w P_exp(w | x) }
    Restricts candidate search space to tokens that are plausible according to the expert,
    preventing the penalty from selecting hallucinations or nonsensical words.

2. Contrastive Objective:
    Score(v) = log P_exp(v | x) - alpha * log P_ama(v | x)   for v in V_head(x)
    Score(v) = -infinity                                     otherwise

3. Token Selection:
    v* = argmax_{v in V_head} Score(v)                      (Greedy Mode)
    or sample from Softmax(Score(v) / tau)                  (Sampling Mode)
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class ContrastiveDecodingStats:
    """Statistics tracking performance and behavior of contrastive decoding."""
    total_tokens_generated: int = 0
    avg_plausibility_set_size: float = 0.0
    num_steps: int = 0
    plausibility_set_sizes: List[int] = None

    def __post_init__(self):
        if self.plausibility_set_sizes is None:
            self.plausibility_set_sizes = []


class ContrastiveDecoder:
    """
    Pure NumPy Contrastive Decoding engine.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        temperature: float = 1.0,
        do_sample: bool = False,
    ):
        """
        Args:
            alpha: Weight of amateur model penalty (higher = stronger amateur suppression)
            beta: Adaptive Plausibility Constraint (APC) cutoff ratio in (0.0, 1.0]
            temperature: Temperature applied to contrastive scores when do_sample=True
            do_sample: If True, sample from contrastive score distribution; otherwise pick argmax
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = max(float(temperature), 1e-5)
        self.do_sample = do_sample

    def _get_probs_and_log_probs(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute numerically stable probability and log-probability distributions."""
        if logits.ndim > 1:
            logits = logits.flatten()
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits) + 1e-12
        probs = exp_logits / sum_exp
        log_probs = (logits - max_logit) - np.log(sum_exp)
        return probs, log_probs

    def compute_contrastive_scores(
        self,
        expert_logits: np.ndarray,
        amateur_logits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute contrastive search scores with Adaptive Plausibility Constraint.

        Args:
            expert_logits: 1D array of logits from the expert model
            amateur_logits: 1D array of logits from the amateur model

        Returns:
            contrastive_scores: 1D array with -inf for unviable candidates
            plausible_indices: 1D boolean mask of valid candidate tokens in V_head
        """
        expert_probs, expert_log_probs = self._get_probs_and_log_probs(expert_logits)
        _, amateur_log_probs = self._get_probs_and_log_probs(amateur_logits)

        # 1. Adaptive Plausibility Constraint: V_head = { v | P_exp(v) >= beta * max_w P_exp(w) }
        max_expert_prob = np.max(expert_probs)
        plausibility_threshold = self.beta * max_expert_prob
        plausible_mask = expert_probs >= plausibility_threshold

        # Ensure at least the top expert token is always included
        if not np.any(plausible_mask):
            plausible_mask[np.argmax(expert_probs)] = True

        # 2. Contrastive objective: S(v) = log P_exp(v) - alpha * log P_ama(v)
        contrastive_scores = np.full_like(expert_log_probs, -np.inf, dtype=np.float32)
        contrastive_scores[plausible_mask] = (
            expert_log_probs[plausible_mask] - self.alpha * amateur_log_probs[plausible_mask]
        )

        return contrastive_scores, plausible_mask

    def select_next_token(
        self,
        expert_logits: np.ndarray,
        amateur_logits: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Select next token ID using contrastive objective.

        Returns:
            selected_token_id: Integer index of selected token
            num_plausible: Count of candidates in plausibility set
        """
        scores, plausible_mask = self.compute_contrastive_scores(expert_logits, amateur_logits)
        num_plausible = int(np.sum(plausible_mask))

        if not self.do_sample:
            # Greedy contrastive selection
            selected_token = int(np.argmax(scores))
            return selected_token, num_plausible

        # Sampling mode: Softmax over plausible contrastive scores
        plausible_scores = scores[plausible_mask] / self.temperature
        max_score = np.max(plausible_scores)
        exp_scores = np.exp(plausible_scores - max_score)
        probs = exp_scores / (np.sum(exp_scores) + 1e-12)

        plausible_indices = np.where(plausible_mask)[0]
        selected_token = int(np.random.choice(plausible_indices, p=probs))
        return selected_token, num_plausible

    def generate(
        self,
        expert_model,
        amateur_model,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[int], ContrastiveDecodingStats]:
        """
        Execute full contrastive autoregressive generation.

        Args:
            expert_model: Competent primary model
            amateur_model: Lightweight draft/amateur model
            prompt_ids: List of input prompt token IDs
            max_new_tokens: Maximum new tokens to generate
            eos_token_id: Optional End-of-Sequence token ID to terminate generation

        Returns:
            generated_ids: List of all tokens (prompt + generated)
            stats: ContrastiveDecodingStats with diagnostic information
        """
        stats = ContrastiveDecodingStats()
        generated = list(prompt_ids)
        expert_context_len = getattr(expert_model, "seq_length", 128)
        amateur_context_len = getattr(amateur_model, "seq_length", 128)

        for _ in range(max_new_tokens):
            # Crop inputs to respective context windows
            expert_input = np.array([generated[-expert_context_len:]], dtype=np.int32)
            amateur_input = np.array([generated[-amateur_context_len:]], dtype=np.int32)

            expert_logits = expert_model.forward(expert_input)[0, -1, :]
            amateur_logits = amateur_model.forward(amateur_input)[0, -1, :]

            next_token, num_plausible = self.select_next_token(expert_logits, amateur_logits)

            generated.append(next_token)
            stats.total_tokens_generated += 1
            stats.num_steps += 1
            stats.plausibility_set_sizes.append(num_plausible)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        if stats.plausibility_set_sizes:
            stats.avg_plausibility_set_size = float(np.mean(stats.plausibility_set_sizes))

        return generated, stats
