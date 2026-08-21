"""
Key-Value (KV) Cache and Optimized Attention Engine

Provides:
- KVCache: Static and dynamic Key-Value tensor cache for O(1) step autoregressive generation
- FastAttention: Vectorized scaled dot-product attention with cache update hooks
"""

from typing import Tuple, Optional, List
import numpy as np


class KVCache:
    """
    Key-Value Cache for accelerating autoregressive transformer inference.
    
    Instead of recomputing key and value representations for all past tokens
    at every single generation step (which is O(N^2) total compute), the KV cache
    stores computed K and V states, reducing per-token step time from O(N) to O(1).
    Supports standard MHA, Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).
    """
    
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, dtype=np.float32):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # Preallocated buffers: (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = np.zeros((max_batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
        self.v_cache = np.zeros((max_batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
        self.current_len = 0
        
    def reset(self):
        """Reset cache position pointer without reallocating buffers."""
        self.current_len = 0
        
    def update(self, key_states: np.ndarray, value_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Append new key and value states into the cache.
        
        Args:
            key_states: shape (batch_size, num_heads, new_seq_len, head_dim)
            value_states: shape (batch_size, num_heads, new_seq_len, head_dim)
            
        Returns:
            Tuple of (all_keys, all_values) spanning sequence positions [0, current_len + new_seq_len)
        """
        batch_size, num_heads, new_len, head_dim = key_states.shape
        start_idx = self.current_len
        end_idx = start_idx + new_len
        
        if end_idx > self.max_seq_len:
            raise ValueError(f"KV Cache overflow: sequence length {end_idx} exceeds max {self.max_seq_len}")
            
        self.k_cache[:batch_size, :num_heads, start_idx:end_idx, :] = key_states
        self.v_cache[:batch_size, :num_heads, start_idx:end_idx, :] = value_states
        self.current_len = end_idx
        
        cached_k = self.k_cache[:batch_size, :num_heads, :end_idx, :]
        cached_v = self.v_cache[:batch_size, :num_heads, :end_idx, :]
        return cached_k, cached_v


class LayerKVCacheManager:
    """Manages separate KV caches across multiple transformer layers."""
    
    def __init__(self, num_layers: int, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.caches = [
            KVCache(max_batch_size, max_seq_len, num_heads, head_dim)
            for _ in range(num_layers)
        ]
        
    def reset(self):
        for cache in self.caches:
            cache.reset()
            
    def get(self, layer_idx: int) -> KVCache:
        return self.caches[layer_idx]
    
    @property
    def current_len(self) -> int:
        return self.caches[0].current_len if self.caches else 0


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized Scaled Dot-Product Attention in NumPy.
    
    Args:
        q: Queries of shape (batch, heads, q_len, d_k)
        k: Keys of shape (batch, heads, kv_len, d_k)
        v: Values of shape (batch, heads, kv_len, d_k)
        mask: Optional additive attention mask (batch or 1, heads or 1, q_len, kv_len)
        scale: Scaling factor (defaults to 1 / sqrt(d_k))
        
    Returns:
        output: shape (batch, heads, q_len, d_k)
        attn_weights: shape (batch, heads, q_len, kv_len)
    """
    d_k = q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)
        
    # Q @ K^T -> shape (batch, heads, q_len, kv_len)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    
    if mask is not None:
        scores = scores + mask
        
    # Stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
    
    # Attention @ V -> shape (batch, heads, q_len, d_k)
    output = np.matmul(attn_weights, v)
    return output, attn_weights


class CachedModernTransformer:
    """
    Complete modern autoregressive transformer model with native KV-cache acceleration.
    
    Combines:
    - Token Embedding lookup
    - Rotary Position Embeddings (RoPE)
    - RMSNorm Pre-Normalization
    - Grouped-Query Attention (GQA) with dynamic KV-caching
    - SwiGLU Feed-Forward Networks
    - Weight-tied LM Head
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = 2,
        ffn_hidden_dim: Optional[int] = None,
        max_seq_len: int = 512,
    ):
        from layers import RMSNorm, ModernTransformerBlock, repeat_kv
        from positional_embeddings import RotaryEmbedding
        from gpt_numpy import Embedding, Linear
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        
        self.token_emb = Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        
        self.blocks = [
            ModernTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                ffn_hidden_dim=ffn_hidden_dim
            )
            for _ in range(num_layers)
        ]
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight.copy()
        
    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Full non-cached forward pass (used for training and prompt prefill).
        
        Args:
            indices: shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = indices.shape
        x = self.token_emb.forward(indices)
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        mask = mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block.forward(x, mask=mask, rope_emb=self.rope, rope_offset=0)
            
        x_norm = self.final_norm.forward(x)
        logits = self.lm_head.forward(x_norm)
        return logits

    def create_cache_manager(self, batch_size: int = 1) -> LayerKVCacheManager:
        """Allocate a new KV cache manager configured for this model."""
        return LayerKVCacheManager(
            num_layers=self.num_layers,
            max_batch_size=batch_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        
    def forward_step(
        self,
        token_indices: np.ndarray,
        cache_mgr: LayerKVCacheManager,
        pos_offset: int
    ) -> np.ndarray:
        """
        Step forward for a single token using preallocated KV-caches.
        
        Args:
            token_indices: shape (batch_size, 1)
            cache_mgr: LayerKVCacheManager instance
            pos_offset: Current sequence position offset
            
        Returns:
            Logits of shape (batch_size, 1, vocab_size)
        """
        from layers import repeat_kv
        
        batch_size, seq_len = token_indices.shape
        assert seq_len == 1, "forward_step expects single token input (seq_len=1)"
        
        x = self.token_emb.forward(token_indices)  # (B, 1, d_model)
        
        for layer_idx, block in enumerate(self.blocks):
            # Pre-norm
            norm1_out = block.norm1.forward(x)
            attn_layer = block.attn
            
            # Linear projections
            q_proj = norm1_out @ attn_layer.W_q  # (B, 1, H * D_h)
            k_proj = norm1_out @ attn_layer.W_k  # (B, 1, H_kv * D_h)
            v_proj = norm1_out @ attn_layer.W_v  # (B, 1, H_kv * D_h)
            
            q = q_proj.reshape(batch_size, 1, attn_layer.num_heads, attn_layer.head_dim).transpose(0, 2, 1, 3)
            k = k_proj.reshape(batch_size, 1, attn_layer.num_kv_heads, attn_layer.head_dim).transpose(0, 2, 1, 3)
            v = v_proj.reshape(batch_size, 1, attn_layer.num_kv_heads, attn_layer.head_dim).transpose(0, 2, 1, 3)
            
            # Apply RoPE at pos_offset
            q = self.rope.apply_rope(q, offset=pos_offset)
            k = self.rope.apply_rope(k, offset=pos_offset)
            
            # Cache update
            layer_cache = cache_mgr.get(layer_idx)
            full_k, full_v = layer_cache.update(k, v)  # (B, H_kv, total_len, D_h)
            
            # Repeat KV heads
            full_k_rep = repeat_kv(full_k, attn_layer.num_queries_per_kv)  # (B, H, total_len, D_h)
            full_v_rep = repeat_kv(full_v, attn_layer.num_queries_per_kv)  # (B, H, total_len, D_h)
            
            # Scaled Dot-Product Attention: Q (1 token) vs full K (total_len tokens)
            scale = 1.0 / np.sqrt(attn_layer.head_dim)
            scores = (q @ full_k_rep.transpose(0, 1, 3, 2)) * scale  # (B, H, 1, total_len)
            
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
            
            context = attn_weights @ full_v_rep  # (B, H, 1, D_h)
            context_flat = context.transpose(0, 2, 1, 3).reshape(batch_size, 1, attn_layer.q_dim)
            attn_out = context_flat @ attn_layer.W_o
            
            # First residual
            h = x + attn_out
            
            # Second pre-norm + SwiGLU FFN + second residual
            norm2_out = block.norm2.forward(h)
            ffn_out = block.ffn.forward(norm2_out)
            x = h + ffn_out
            
        x_norm = self.final_norm.forward(x)
        logits = self.lm_head.forward(x_norm)
        return logits

    def generate_cached(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        sampler: Optional[object] = None,
        stop_token_ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Fast autoregressive text generation using KV-cache.
        
        Args:
            prompt_ids: Initial token ID sequence
            max_new_tokens: Maximum number of new tokens to generate
            sampler: GenerationSampler instance (defaults to greedy search)
            stop_token_ids: Optional list of token IDs to terminate generation
            
        Returns:
            Full generated token ID list (prompt + generated tokens)
        """
        from sampler import GenerationSampler
        sampler = sampler or GenerationSampler(temperature=0.0)
        stop_token_ids = stop_token_ids or []
        
        generated = list(prompt_ids)
        cache_mgr = self.create_cache_manager(batch_size=1)
        
        # Step 1: Prefill prompt tokens one by one (or in prefill pass)
        for i, tok in enumerate(prompt_ids):
            token_arr = np.array([[tok]], dtype=np.int32)
            logits = self.forward_step(token_arr, cache_mgr, pos_offset=i)
            
        last_logits = logits[0, -1, :]
        next_token = sampler.sample_token(last_logits, generated_ids=generated)
        
        if next_token in stop_token_ids:
            return generated
            
        generated.append(next_token)
        
        # Step 2: Step-by-step token generation with O(1) compute per step
        for step in range(len(prompt_ids), len(prompt_ids) + max_new_tokens - 1):
            token_arr = np.array([[generated[-1]]], dtype=np.int32)
            logits = self.forward_step(token_arr, cache_mgr, pos_offset=step)
            last_logits = logits[0, -1, :]
            
            next_token = sampler.sample_token(last_logits, generated_ids=generated)
            if next_token in stop_token_ids:
                break
                
            generated.append(next_token)
            
        return generated

