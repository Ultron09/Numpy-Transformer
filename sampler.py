"""
Generation Sampler & Decoding Strategies for Autoregressive Transformers

Provides pure NumPy implementations of:
- Greedy Search (argmax)
- Temperature Scaling
- Top-K Sampling
- Top-P (Nucleus) Sampling
- Min-P Sampling
- Repetition & Frequency Penalties
- Streaming generation generator
"""

from typing import List, Optional, Callable, Union
import numpy as np


class GenerationSampler:
    """
    Advanced text generation sampler supporting modern LLM decoding heuristics.
    """
    
    def __init__(
        self,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repetition_penalty: float = 1.15,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        """
        Args:
            temperature: Sampling temperature (lower = more deterministic, higher = more creative)
            top_k: Keep only top K highest probability tokens (0 to disable)
            top_p: Nucleus sampling probability mass threshold (1.0 to disable)
            min_p: Minimum probability threshold relative to the most likely token (0.0 to disable)
            repetition_penalty: Multiplicative penalty applied to previously generated tokens (1.0 = none)
            frequency_penalty: Additive penalty proportional to token frequency in generated sequence
            presence_penalty: Additive penalty for any token already present in generated sequence
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
    
    def apply_repetition_penalties(
        self,
        logits: np.ndarray,
        generated_ids: List[int]
    ) -> np.ndarray:
        """
        Apply multiplicative and additive repetition penalties on logits.
        """
        if not generated_ids:
            return logits.copy()
            
        modified_logits = logits.copy()
        unique_tokens, counts = np.unique(generated_ids, return_counts=True)
        token_count_map = dict(zip(unique_tokens, counts))
        
        for token_id, count in token_count_map.items():
            if 0 <= token_id < len(modified_logits):
                # Multiplicative repetition penalty (CTRL paper)
                if self.repetition_penalty != 1.0:
                    if modified_logits[token_id] > 0:
                        modified_logits[token_id] /= self.repetition_penalty
                    else:
                        modified_logits[token_id] *= self.repetition_penalty
                        
                # Additive frequency and presence penalties (OpenAI style)
                modified_logits[token_id] -= (self.frequency_penalty * count + self.presence_penalty)
                
        return modified_logits
    
    def sample_token(
        self,
        logits: np.ndarray,
        generated_ids: Optional[List[int]] = None
    ) -> int:
        """
        Sample a next token ID from raw unnormalized logits using configured strategies.
        
        Args:
            logits: 1D NumPy array of shape (vocab_size,)
            generated_ids: List of previously generated token IDs for penalties
            
        Returns:
            Selected integer token index
        """
        if logits.ndim > 1:
            logits = logits.flatten()
            
        # 1. Apply repetition penalties
        if generated_ids is not None and len(generated_ids) > 0:
            logits = self.apply_repetition_penalties(logits, generated_ids)
            
        # 2. Greedy search if temperature <= 0
        if self.temperature <= 1e-6:
            return int(np.argmax(logits))
            
        # 3. Temperature scaling
        logits = logits / self.temperature
        
        # 4. Numerically stable softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
        
        # 5. Top-K filtering
        if self.top_k > 0 and self.top_k < len(probs):
            top_k_indices = np.argpartition(probs, -self.top_k)[-self.top_k:]
            mask = np.ones(len(probs), dtype=bool)
            mask[top_k_indices] = False
            probs[mask] = 0.0
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.zeros_like(probs)
                probs[top_k_indices] = 1.0 / len(top_k_indices)
                
        # 6. Min-P filtering
        if self.min_p > 0.0:
            max_prob = np.max(probs)
            min_thresh = self.min_p * max_prob
            probs[probs < min_thresh] = 0.0
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum
                
        # 7. Top-P (Nucleus) filtering
        if 0.0 < self.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Identify tokens to remove (cumulative prob exceeds threshold)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Keep at least the first token
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.zeros_like(probs)
                probs[sorted_indices[0]] = 1.0
                
        # Final distribution check
        probs = np.nan_to_num(probs, nan=0.0)
        prob_sum = np.sum(probs)
        if prob_sum <= 0:
            return int(np.argmax(logits))
        probs = probs / prob_sum
        
        # Draw categorical sample
        return int(np.random.choice(len(probs), p=probs))


def generate_stream(
    model,
    prompt_ids: List[int],
    max_new_tokens: int = 50,
    sampler: Optional[GenerationSampler] = None,
    stop_token_ids: Optional[List[int]] = None,
    callback: Optional[Callable[[int], None]] = None
):
    """
    Autoregressive generator yielding newly generated token IDs one-by-one.
    """
    sampler = sampler or GenerationSampler()
    stop_token_ids = stop_token_ids or []
    generated = list(prompt_ids)
    
    for _ in range(max_new_tokens):
        # Crop context to model sequence length if needed
        context_len = getattr(model, "seq_length", 128)
        input_tokens = generated[-context_len:]
        input_arr = np.array([input_tokens], dtype=np.int32)
        
        # Forward pass to get logits: shape (batch_size, seq_len, vocab_size)
        logits = model.forward(input_arr, training=False)
        last_logits = logits[0, -1, :]
        
        next_token = sampler.sample_token(last_logits, generated_ids=generated)
        
        if next_token in stop_token_ids:
            break
            
        generated.append(next_token)
        if callback:
            callback(next_token)
            
        yield next_token
