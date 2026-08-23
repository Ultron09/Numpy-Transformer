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

from typing import List, Tuple, Optional, Callable, Union
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


class BeamHypothesis:
    """
    Represents a single hypothesis branch during beam search decoding.
    """
    def __init__(self, tokens: List[int], log_prob: float = 0.0):
        self.tokens = list(tokens)
        self.log_prob = log_prob
        self.is_done = False
        
    @property
    def length(self) -> int:
        return len(self.tokens)
        
    def compute_score(self, length_penalty: float = 1.0, prompt_len: int = 0) -> float:
        """
        Compute length-normalized log-probability score using GNMT length penalty formula:
            score = log_prob / ((5 + gen_len) / (5 + 1))^length_penalty
        """
        gen_len = max(1, len(self.tokens) - prompt_len)
        if length_penalty == 0.0:
            return self.log_prob
        lp = ((5.0 + gen_len) / 6.0) ** length_penalty
        return self.log_prob / lp

    def __repr__(self) -> str:
        return f"BeamHypothesis(tokens={self.tokens}, log_prob={self.log_prob:.4f}, done={self.is_done})"


class BeamSearchDecoder:
    """
    Beam Search Decoder with length normalization, repetition penalties, and n-gram blocking.
    
    Maintains the top-B most probable hypothesis paths through the vocabulary space,
    significantly improving sequence coherence and output quality over greedy decoding.
    """
    
    def __init__(
        self,
        beam_width: int = 4,
        max_new_tokens: int = 50,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = True,
    ):
        """
        Args:
            beam_width: Number of active candidate beams tracked per decoding step
            max_new_tokens: Maximum number of tokens to generate
            length_penalty: Exponential penalty factor for sequence length (1.0 = standard, >1.0 favors longer)
            no_repeat_ngram_size: If > 0, forbids generating previously generated n-grams of this length
            repetition_penalty: Multiplicative repetition penalty applied to logits
            early_stopping: Whether to terminate when beam_width completed hypotheses are found
        """
        self.beam_width = beam_width
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.early_stopping = early_stopping

    def _block_ngrams(self, sequence: List[int], logits: np.ndarray) -> np.ndarray:
        """
        Set logits of tokens that would complete an already seen n-gram to -inf.
        """
        if self.no_repeat_ngram_size <= 0 or len(sequence) < self.no_repeat_ngram_size:
            return logits
            
        n = self.no_repeat_ngram_size
        prefix = tuple(sequence[-(n - 1):])
        
        # Build dictionary of seen n-grams
        seen_continuations = set()
        for i in range(len(sequence) - n + 1):
            if tuple(sequence[i:i + n - 1]) == prefix:
                seen_continuations.add(sequence[i + n - 1])
                
        if seen_continuations:
            logits = logits.copy()
            for token_id in seen_continuations:
                if 0 <= token_id < len(logits):
                    logits[token_id] = -1e9
                    
        return logits

    def _apply_penalties(self, sequence: List[int], logits: np.ndarray) -> np.ndarray:
        """
        Apply repetition penalty and n-gram blocking.
        """
        logits = logits.copy()
        if self.repetition_penalty != 1.0 and sequence:
            for token_id in set(sequence):
                if 0 <= token_id < len(logits):
                    if logits[token_id] > 0:
                        logits[token_id] /= self.repetition_penalty
                    else:
                        logits[token_id] *= self.repetition_penalty
                        
        return self._block_ngrams(sequence, logits)

    def search(
        self,
        model,
        prompt_ids: List[int],
        eos_token_id: Optional[int] = None,
        num_return_sequences: int = 1,
    ) -> List[Tuple[List[int], float]]:
        """
        Execute beam search over the autoregressive model.
        
        Args:
            model: Transformer model exposing .forward(tokens_arr) -> logits
            prompt_ids: List of integer prompt token IDs
            eos_token_id: Optional End-of-Sequence token ID
            num_return_sequences: Number of top scoring sequences to return
            
        Returns:
            List of tuples: (generated_token_ids, normalized_score) sorted descending by score
        """
        prompt_len = len(prompt_ids)
        beams = [BeamHypothesis(tokens=prompt_ids, log_prob=0.0)]
        completed_hypotheses: List[BeamHypothesis] = []
        
        context_len = getattr(model, "seq_length", 128)
        
        for step in range(self.max_new_tokens):
            candidates = []
            
            # Step 1: Forward pass for each active beam
            for beam in beams:
                if beam.is_done:
                    candidates.append(beam)
                    continue
                    
                input_tokens = beam.tokens[-context_len:]
                input_arr = np.array([input_tokens], dtype=np.int32)
                
                # Get logits for the next token
                logits = model.forward(input_arr)[0, -1, :]
                
                # Apply penalties
                logits = self._apply_penalties(beam.tokens, logits)
                
                # Compute log softmax: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
                max_logit = np.max(logits)
                exp_logits = np.exp(logits - max_logit)
                log_probs = (logits - max_logit) - np.log(np.sum(exp_logits) + 1e-12)
                
                # Extract top 2 * beam_width candidates for this beam
                top_indices = np.argpartition(log_probs, -min(2 * self.beam_width, len(log_probs)))[-min(2 * self.beam_width, len(log_probs)):]
                top_indices = top_indices[np.argsort(log_probs[top_indices])[::-1]]
                
                for token_id in top_indices:
                    token_log_prob = log_probs[token_id]
                    if token_log_prob < -1e8:
                        continue
                    new_tokens = beam.tokens + [int(token_id)]
                    new_hyp = BeamHypothesis(tokens=new_tokens, log_prob=beam.log_prob + float(token_log_prob))
                    
                    if eos_token_id is not None and token_id == eos_token_id:
                        new_hyp.is_done = True
                        completed_hypotheses.append(new_hyp)
                    else:
                        candidates.append(new_hyp)
                        
            # Step 2: Prune candidates to top beam_width by current score
            if not candidates:
                break
                
            candidates.sort(key=lambda h: h.compute_score(self.length_penalty, prompt_len), reverse=True)
            beams = candidates[:self.beam_width]
            
            # Step 3: Check early stopping criteria
            if self.early_stopping and len(completed_hypotheses) >= self.beam_width:
                break
                
        # Combine remaining active beams with completed hypotheses
        all_hypotheses = completed_hypotheses + [b for b in beams if not b.is_done]
        if not all_hypotheses:
            all_hypotheses = beams
            
        # Rank by final length-normalized score
        all_hypotheses.sort(key=lambda h: h.compute_score(self.length_penalty, prompt_len), reverse=True)
        
        results = []
        for hyp in all_hypotheses[:num_return_sequences]:
            score = hyp.compute_score(self.length_penalty, prompt_len)
            results.append((hyp.tokens, float(score)))
            
        return results


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
        logits = model.forward(input_arr)
        last_logits = logits[0, -1, :]
        
        next_token = sampler.sample_token(last_logits, generated_ids=generated)
        
        if next_token in stop_token_ids:
            break
            
        generated.append(next_token)
        if callback:
            callback(next_token)
            
        yield next_token
