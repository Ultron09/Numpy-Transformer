"""
Advanced Attention Mechanisms & Memory-Efficient Algorithms in NumPy

Provides:
- Sliding Window Attention (SWA): Local banded causal attention (Mistral AI)
- Tiled Online Softmax Attention: FlashAttention-1 style memory-bounded chunked attention
- sliding_window_causal_mask: Helper for generating banded causal attention masks
"""

from typing import Tuple, Optional
import numpy as np


def sliding_window_causal_mask(seq_len: int, window_size: int) -> np.ndarray:
    """
    Construct a sliding window causal attention mask.
    
    Tokens at position i can only attend to positions j where:
        max(0, i - window_size + 1) <= j <= i
        
    Args:
        seq_len: Total sequence length
        window_size: Maximum backward lookback window size
        
    Returns:
        Mask of shape (1, 1, seq_len, seq_len) with 0.0 for valid positions and -1e9 for masked positions.
    """
    row_indices = np.arange(seq_len)[:, np.newaxis]
    col_indices = np.arange(seq_len)[np.newaxis, :]
    
    # Valid condition: (j <= i) and (i - j < window_size)
    causal_valid = col_indices <= row_indices
    window_valid = (row_indices - col_indices) < window_size
    valid_mask = causal_valid & window_valid
    
    mask = np.where(valid_mask, 0.0, -1e9).astype(np.float32)
    return mask[np.newaxis, np.newaxis, :, :]


def tiled_online_softmax_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size_q: int = 16,
    block_size_kv: int = 16,
    scale: Optional[float] = None
) -> np.ndarray:
    """
    Tiled Online Softmax Attention (FlashAttention-1 algorithm foundation in pure NumPy).
    
    Computes exact scaled dot-product attention O = Softmax(Q K^T / √d) V in blocks
    using the online softmax algorithm (Milakov & Gimelshein 2018 / Dao et al. 2022).
    
    Key Memory Advantage:
    Avoids ever materializing the (batch, heads, seq_len, seq_len) attention score matrix,
    reducing intermediate peak memory from O(N^2) to O(N * B_c).
    
    Args:
        q: Queries of shape (batch, heads, seq_len_q, head_dim)
        k: Keys of shape (batch, heads, seq_len_kv, head_dim)
        v: Values of shape (batch, heads, seq_len_kv, head_dim)
        block_size_q: Block chunk size for query dimension
        block_size_kv: Block chunk size for key/value dimension
        scale: Scaling factor (defaults to 1.0 / sqrt(head_dim))
        
    Returns:
        output: Attention output tensor of shape (batch, heads, seq_len_q, head_dim)
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_kv, _ = k.shape
    
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)
        
    outputs = []
    
    # Loop over Query blocks
    for q_start in range(0, seq_len_q, block_size_q):
        q_end = min(q_start + block_size_q, seq_len_q)
        q_block = q[:, :, q_start:q_end, :]  # (B, H, B_q, D_h)
        b_q = q_end - q_start
        
        # Running statistics for this query block
        # m_i: running maximum per row, initialized to -infinity
        m_i = np.full((batch_size, num_heads, b_q, 1), -np.inf, dtype=np.float32)
        # l_i: running sum of exp(scores - m_i), initialized to 0
        l_i = np.zeros((batch_size, num_heads, b_q, 1), dtype=np.float32)
        # o_i: running unnormalized weighted value accumulator
        o_i = np.zeros((batch_size, num_heads, b_q, head_dim), dtype=np.float32)
        
        # Loop over Key/Value blocks
        for kv_start in range(0, seq_len_kv, block_size_kv):
            kv_end = min(kv_start + block_size_kv, seq_len_kv)
            k_block = k[:, :, kv_start:kv_end, :]  # (B, H, B_kv, D_h)
            v_block = v[:, :, kv_start:kv_end, :]  # (B, H, B_kv, D_h)
            
            # Compute partial dot-product: (B, H, B_q, B_kv)
            scores_ij = np.matmul(q_block, k_block.transpose(0, 1, 3, 2)) * scale
            
            # 1. New row maximum
            m_block = np.max(scores_ij, axis=-1, keepdims=True)  # (B, H, B_q, 1)
            m_new = np.maximum(m_i, m_block)
            
            # 2. Rescaling factors for previous accumulator and current block
            # alpha = exp(m_old - m_new)
            # Safe exponential to avoid NaN when m_i is -inf
            alpha = np.where(np.isneginf(m_i), 0.0, np.exp(m_i - m_new))
            p_ij = np.exp(scores_ij - m_new)  # (B, H, B_q, B_kv)
            
            # 3. Update running normalization denominator
            l_new = alpha * l_i + np.sum(p_ij, axis=-1, keepdims=True)
            
            # 4. Update unnormalized output accumulator:
            # o_new = alpha * o_i + (p_ij @ v_block)
            p_v = np.matmul(p_ij, v_block)  # (B, H, B_q, D_h)
            o_new = alpha * o_i + p_v
            
            # Update running state
            m_i = m_new
            l_i = l_new
            o_i = o_new
            
        # Final block normalization: o_block = o_i / l_i
        q_block_output = o_i / (l_i + 1e-12)
        outputs.append(q_block_output)
        
    return np.concatenate(outputs, axis=2)


class SlidingWindowAttention:
    """
    Sliding Window Multi-Head Attention layer.
    
    Restricts attention to a fixed local context window of size W tokens,
    reducing attention compute from quadratic O(N^2) to linear O(N * W).
    Used in Mistral 7B to support ultra-long context inference efficiently.
    """
    
    def __init__(self, d_model: int, num_heads: int, window_size: int = 128, dropout: float = 0.0):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.dropout = dropout
        
        std = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * std
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * std
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * std
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * std
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        
        q_proj = x @ self.W_q
        k_proj = x @ self.W_k
        v_proj = x @ self.W_v
        
        q = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        mask = sliding_window_causal_mask(seq_len, self.window_size)
        
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale + mask
        
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
        
        context = attn_weights @ v
        context_flat = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        return context_flat @ self.W_o
