"""
Positional Embedding Modules for Transformer Architectures

Includes:
- SinusoidalEmbedding: Fixed trigonometric positional encodings (Vaswani et al.)
- RotaryEmbedding (RoPE): Rotary Position Embedding via 2D vector rotation (Su et al.)
- ALiBiEmbedding: Attention with Linear Biases for length extrapolation (Press et al.)
- LearnedPositionalEmbedding: Standard learned 1D embedding lookup table
"""

from typing import Tuple, Optional
import numpy as np


class SinusoidalEmbedding:
    """
    Fixed Sinusoidal Positional Encoding from 'Attention Is All You Need'.
    
    Mathematical Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    
    def __init__(self, max_seq_len: int, d_model: int):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pe = self._build_pe(max_seq_len, d_model)
        
    def _build_pe(self, max_seq_len: int, d_model: int) -> np.ndarray:
        pe = np.zeros((max_seq_len, d_model), dtype=np.float32)
        position = np.arange(0, max_seq_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
        
    def forward(self, seq_len: int, offset: int = 0) -> np.ndarray:
        """Return positional embeddings for positions [offset, offset + seq_len)."""
        return self.pe[offset:offset + seq_len, :]


class RotaryEmbedding:
    """
    Rotary Position Embedding (RoPE).
    
    Encodes absolute positions with a rotation matrix and naturally incorporates
    relative position dependency in self-attention:
        <R_m q, R_n k> = q^T R_{n-m} k
        
    Used in modern LLMs including LLaMA, Mistral, Gemma, and DeepSeek.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        """
        Args:
            dim: Dimension per head (must be even)
            max_seq_len: Maximum sequence length precomputed
            base: Base for geometric progression of frequencies
        """
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequency: theta_i = 1 / (base ^ (2i / dim))
        self.inv_freq = 1.0 / (self.base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self._cos_cached = None
        self._sin_cached = None
        self._precompute_cache(max_seq_len)
        
    def _precompute_cache(self, seq_len: int):
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self.inv_freq)  # (seq_len, dim/2)
        # Duplicate frequencies to match full head dim
        emb = np.concatenate([freqs, freqs], axis=-1)  # (seq_len, dim)
        self._cos_cached = np.cos(emb)  # (seq_len, dim)
        self._sin_cached = np.sin(emb)  # (seq_len, dim)
        
    def _rotate_half(self, x: np.ndarray) -> np.ndarray:
        """Rotate vector half: [-x2, x1] where x = [x1, x2]."""
        d_2 = x.shape[-1] // 2
        x1 = x[..., :d_2]
        x2 = x[..., d_2:]
        return np.concatenate([-x2, x1], axis=-1)
        
    def apply_rope(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Apply RoPE transformation to query or key tensor.
        
        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, head_dim)
            offset: Starting sequence position offset (useful for KV cache generation)
            
        Returns:
            Rotated tensor of same shape as x
        """
        seq_len = x.shape[2]
        total_len = offset + seq_len
        if total_len > self.max_seq_len:
            self._precompute_cache(total_len * 2)
            self.max_seq_len = total_len * 2
            
        cos = self._cos_cached[offset:offset + seq_len, :]  # (seq_len, dim)
        sin = self._sin_cached[offset:offset + seq_len, :]  # (seq_len, dim)
        
        # Reshape for broadcasting over (batch, heads, seq_len, dim)
        cos = cos[np.newaxis, np.newaxis, :, :]
        sin = sin[np.newaxis, np.newaxis, :, :]
        
        # RoPE: (x * cos) + (rotate_half(x) * sin)
        return (x * cos) + (self._rotate_half(x) * sin)
        
    def backward_rope(self, grad_output: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        Backward pass for RoPE.
        Since rotation matrix R is orthogonal (R^T = R^(-1)), the gradient
        with respect to input is simply the reverse rotation (applying -sin).
        """
        seq_len = grad_output.shape[2]
        cos = self._cos_cached[offset:offset + seq_len, :][np.newaxis, np.newaxis, :, :]
        sin = self._sin_cached[offset:offset + seq_len, :][np.newaxis, np.newaxis, :, :]
        
        # (grad * cos) + rotate_half(grad) * (-sin)
        return (grad_output * cos) - (self._rotate_half(grad_output) * sin)


class ALiBiEmbedding:
    """
    Attention with Linear Biases (ALiBi).
    
    Instead of adding positional embeddings to word representations,
    ALiBi adds static linear distance biases directly to the attention matrix:
        scores = (Q K^T / √d_k) - m * |i - j|
        
    Where m is a geometric head-specific slope: m = 2^(-8/num_heads * head_index).
    Enables remarkable zero-shot sequence length extrapolation.
    """
    
    def __init__(self, num_heads: int):
        self.num_heads = num_heads
        self.slopes = self._get_slopes(num_heads)
        
    def _get_slopes(self, n: int) -> np.ndarray:
        """Compute ALiBi geometric slope sequence."""
        def get_slopes_power_of_2(n_heads: int):
            start = (2 ** (-2 ** -(np.log2(n_heads) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n_heads)]
            
        if np.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** int(np.floor(np.log2(n)))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )
        return np.array(slopes, dtype=np.float32)
        
    def get_bias(self, seq_len: int) -> np.ndarray:
        """
        Generate ALiBi bias matrix of shape (1, num_heads, seq_len, seq_len).
        """
        # Distance matrix |i - j| for causal attention: (j - i)
        positions = np.arange(seq_len)
        distance = positions[np.newaxis, :] - positions[:, np.newaxis]  # (seq_len, seq_len)
        distance = np.abs(distance)
        
        # Slopes shape: (num_heads, 1, 1)
        slopes = self.slopes[:, np.newaxis, np.newaxis]
        
        # Negative linear penalty
        alibi_bias = -(slopes * distance[np.newaxis, :, :])  # (num_heads, seq_len, seq_len)
        return alibi_bias[np.newaxis, :, :, :]  # (1, num_heads, seq_len, seq_len)
