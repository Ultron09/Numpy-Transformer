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
