"""
Modern Transformer Architectural Layers

Includes:
- RMSNorm: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
- SiLU / Swish: Sigmoid Linear Unit activation with analytical gradient
- SwiGLU: Swish-Gated Linear Unit feed-forward layer (Shazeer, 2020)
- ModernTransformerBlock: Pre-LN block combining RMSNorm, RoPE, and SwiGLU
"""

from typing import Tuple, Optional, List
import numpy as np


class RMSNorm:
    """
    Root Mean Square Normalization (RMSNorm).
    
    Replaces standard LayerNorm by normalizing inputs based on the root mean square
    rather than computing both mean and variance. This provides comparable training
    stability with ~10-30% computational savings.
    
    Mathematical Formulation:
        RMS(x) = sqrt(1/d * sum(x_i^2) + eps)
        y = (x / RMS(x)) * gamma
        
    Used by modern LLMs (LLaMA, Mistral, Gemma, Falcon).
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)  # Learned gain parameter
        self.grad_gamma = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input array of shape (..., d_model)
            
        Returns:
            Normalized output of shape (..., d_model)
        """
        self.x = x
        # Compute RMS across last feature dimension: sqrt(mean(x^2) + eps)
        self.rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        self.x_norm = x / self.rms
        return self.gamma * self.x_norm
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for RMSNorm.
        
        Derivation:
            d_gamma = sum(grad_output * x_norm)
            dx = (grad_output * gamma) / rms - (x * sum(grad_output * gamma * x)) / (d * rms^3)
        """
        d = self.d_model
        # Gradient for gamma parameter
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=tuple(range(len(grad_output.shape) - 1)))
        
        # Intermediate gradient w.r.t normalized x
        grad_norm = grad_output * self.gamma
        
        # Full gradient w.r.t input x
        sum_grad_x = np.sum(grad_norm * self.x, axis=-1, keepdims=True)
        grad_x = (grad_norm / self.rms) - (self.x * sum_grad_x / (d * (self.rms ** 3)))
        return grad_x


class SiLU:
    """
    Sigmoid Linear Unit (SiLU / Swish) activation function.
    
    Mathematical Formula:
        SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    Derivative:
        d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                     = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.sigmoid_x = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
        self.out = x * self.sigmoid_x
        return self.out
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        dx = self.sigmoid_x * (1.0 + self.x * (1.0 - self.sigmoid_x))
        return grad_output * dx


class SwiGLU:
    """
    Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.
    
    Replaces standard 2-layer MLP with a gated architecture:
        SwiGLU(x) = (SiLU(x W_gate) ⊙ (x W_up)) W_down
        
    Introduced by Noam Shazeer (2020) and used across all frontier open-weight models.
    """
    
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        if hidden_dim is None:
            # Common 8/3 hidden dim heuristic rounded to multiple of 64
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64
            
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        # Xavier/He initialization
        std = np.sqrt(2.0 / d_model)
        self.W_gate = np.random.randn(d_model, hidden_dim).astype(np.float32) * std
        self.W_up = np.random.randn(d_model, hidden_dim).astype(np.float32) * std
        self.W_down = np.random.randn(hidden_dim, d_model).astype(np.float32) * (np.sqrt(2.0 / hidden_dim))
        
        self.silu = SiLU()
        
        self.grad_W_gate = None
        self.grad_W_up = None
        self.grad_W_down = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        """
        self.x = x
        self.gate_linear = x @ self.W_gate
        self.up_linear = x @ self.W_up
        
        self.gate_activated = self.silu.forward(self.gate_linear)
        self.hidden_state = self.gate_activated * self.up_linear  # Element-wise gate
        
        output = self.hidden_state @ self.W_down
        return output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Analytical backpropagation through SwiGLU gating mechanism.
        """
        # 1. Gradient w.r.t W_down
        # Reshape to 2D matrices for dot products
        h_flat = self.hidden_state.reshape(-1, self.hidden_dim)
        grad_out_flat = grad_output.reshape(-1, self.d_model)
        x_flat = self.x.reshape(-1, self.d_model)
        
        self.grad_W_down = h_flat.T @ grad_out_flat
        
        # 2. Gradient w.r.t hidden_state
        grad_hidden = grad_output @ self.W_down.T  # (..., hidden_dim)
        
        # 3. Product rule on gate * up
        grad_up_linear = grad_hidden * self.gate_activated
        grad_gate_activated = grad_hidden * self.up_linear
        
        # 4. Backprop through SiLU
        grad_gate_linear = self.silu.backward(grad_gate_activated)
        
        # 5. Gradients w.r.t W_up and W_gate
        grad_up_flat = grad_up_linear.reshape(-1, self.hidden_dim)
        grad_gate_flat = grad_gate_linear.reshape(-1, self.hidden_dim)
        
        self.grad_W_up = x_flat.T @ grad_up_flat
        self.grad_W_gate = x_flat.T @ grad_gate_flat
        
        # 6. Gradient w.r.t input x
        grad_x = (grad_gate_linear @ self.W_gate.T) + (grad_up_linear @ self.W_up.T)
        return grad_x


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """
    Repeat key/value heads along the head dimension for Grouped-Query Attention.
    
    Args:
        x: Array of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head (num_heads // num_kv_heads)
        
    Returns:
        Array of shape (batch, num_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)


def unrepeat_kv_grad(grad: np.ndarray, n_rep: int) -> np.ndarray:
    """
    Sum gradients across repeated heads for backward pass of repeat_kv.
    
    Args:
        grad: Gradient array of shape (batch, num_heads, seq_len, head_dim)
        n_rep: Number of times heads were repeated
        
    Returns:
        Gradient array of shape (batch, num_kv_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return grad
    batch, num_heads, seq_len, head_dim = grad.shape
    num_kv_heads = num_heads // n_rep
    reshaped = grad.reshape(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return np.sum(reshaped, axis=2)


class GroupedQueryAttention:
    """
    Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).
    
    Mathematical Formulation:
        For num_heads (H) queries and num_kv_heads (H_kv) keys/values:
        - When H_kv == H: Standard Multi-Head Attention (MHA)
        - When H_kv == 1: Multi-Query Attention (MQA - Shazeer, 2019)
        - When 1 < H_kv < H: Grouped-Query Attention (GQA - Ainslie et al., 2023)
        
    GQA partitions H query heads into H_kv groups of size (H / H_kv).
    Keys and values are shared within each group, drastically reducing KV cache size
    and memory bandwidth during autoregressive decoding while preserving full MHA quality.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        self.head_dim = head_dim if head_dim is not None else (d_model // num_heads)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.dropout = dropout
        
        # Xavier/He initialization
        q_std = np.sqrt(2.0 / (d_model + self.q_dim))
        kv_std = np.sqrt(2.0 / (d_model + self.kv_dim))
        out_std = np.sqrt(2.0 / (self.q_dim + d_model))
        
        self.W_q = np.random.randn(d_model, self.q_dim).astype(np.float32) * q_std
        self.W_k = np.random.randn(d_model, self.kv_dim).astype(np.float32) * kv_std
        self.W_v = np.random.randn(d_model, self.kv_dim).astype(np.float32) * kv_std
        self.W_o = np.random.randn(self.q_dim, d_model).astype(np.float32) * out_std
        
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None
        
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rope_emb: Optional[object] = None,
        rope_offset: int = 0
    ) -> np.ndarray:
        """
        Forward pass for Grouped-Query Attention.
        
        Args:
            x: Input array of shape (batch, seq_len, d_model)
            mask: Optional attention mask (additive, -inf for masked entries)
            rope_emb: Optional RotaryEmbedding instance to apply
            rope_offset: Sequence offset index for RoPE
            
        Returns:
            Output array of shape (batch, seq_len, d_model)
        """
        self.x = x
        batch_size, seq_len, _ = x.shape
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rope_emb = rope_emb
        self.rope_offset = rope_offset
        
        # 1. Linear projections
        q_proj = x @ self.W_q  # (B, L, H * D_h)
        k_proj = x @ self.W_k  # (B, L, H_kv * D_h)
        v_proj = x @ self.W_v  # (B, L, H_kv * D_h)
        
        # 2. Reshape to multi-head tensors
        q = q_proj.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k_proj.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_proj.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 3. Apply RoPE if provided
        if rope_emb is not None:
            q = rope_emb.apply_rope(q, offset=rope_offset)
            k = rope_emb.apply_rope(k, offset=rope_offset)
            
        self.q = q
        self.k = k
        self.v = v
        
        # 4. Repeat KV heads to match query heads
        k_rep = repeat_kv(k, self.num_queries_per_kv)  # (B, H, L, D_h)
        v_rep = repeat_kv(v, self.num_queries_per_kv)  # (B, H, L, D_h)
        self.k_rep = k_rep
        self.v_rep = v_rep
        
        # 5. Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k_rep.transpose(0, 1, 3, 2)) * scale  # (B, H, L_q, L_k)
        
        if mask is not None:
            scores = scores + mask
            
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)
        self.attn_weights = attn_weights
        
        # 6. Context projection
        context = attn_weights @ v_rep  # (B, H, L, D_h)
        self.context = context
        
        # Reshape to (B, L, H * D_h)
        self.context_flat = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.q_dim)
        
        # 7. Output projection
        out = self.context_flat @ self.W_o
        return out
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Analytical backpropagation through Grouped-Query Attention.
        """
        batch_size, seq_len = self.batch_size, self.seq_len
        
        # 1. Output projection gradients
        ctx_2d = self.context_flat.reshape(-1, self.q_dim)
        grad_out_2d = grad_output.reshape(-1, self.d_model)
        self.grad_W_o = ctx_2d.T @ grad_out_2d
        
        grad_context_flat = grad_output @ self.W_o.T  # (B, L, q_dim)
        grad_context = grad_context_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 2. Context = attn_weights @ v_rep
        grad_v_rep = self.attn_weights.transpose(0, 1, 3, 2) @ grad_context  # (B, H, L, D_h)
        grad_attn_weights = grad_context @ self.v_rep.transpose(0, 1, 3, 2)  # (B, H, L_q, L_k)
        
        # 3. Softmax backward
        sum_grad_attn = np.sum(grad_attn_weights * self.attn_weights, axis=-1, keepdims=True)
        grad_scores = self.attn_weights * (grad_attn_weights - sum_grad_attn)
        
        # 4. Scores = (q @ k_rep.T) * scale
        scale = 1.0 / np.sqrt(self.head_dim)
        grad_q = (grad_scores @ self.k_rep) * scale  # (B, H, L, D_h)
        grad_k_rep = (grad_scores.transpose(0, 1, 3, 2) @ self.q) * scale  # (B, H, L, D_h)
        
        # 5. Reverse RoPE if applied
        if self.rope_emb is not None:
            grad_q = self.rope_emb.backward_rope(grad_q, offset=self.rope_offset)
            grad_k_rep = self.rope_emb.backward_rope(grad_k_rep, offset=self.rope_offset)
            
        # 6. Unrepeat KV gradients
        grad_k = unrepeat_kv_grad(grad_k_rep, self.num_queries_per_kv)  # (B, H_kv, L, D_h)
        grad_v = unrepeat_kv_grad(grad_v_rep, self.num_queries_per_kv)  # (B, H_kv, L, D_h)
        
        # 7. Reshape to linear projection shapes
        grad_q_proj = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.q_dim)
        grad_k_proj = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.kv_dim)
        grad_v_proj = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.kv_dim)
        
        x_2d = self.x.reshape(-1, self.d_model)
        self.grad_W_q = x_2d.T @ grad_q_proj.reshape(-1, self.q_dim)
        self.grad_W_k = x_2d.T @ grad_k_proj.reshape(-1, self.kv_dim)
        self.grad_W_v = x_2d.T @ grad_v_proj.reshape(-1, self.kv_dim)
        
        # 8. Gradient w.r.t input x
        grad_x = (
            (grad_q_proj @ self.W_q.T) +
            (grad_k_proj @ self.W_k.T) +
            (grad_v_proj @ self.W_v.T)
        )
        return grad_x


class ModernTransformerBlock:
    """
    Pre-LN Modern Transformer Block combining:
    - RMSNorm pre-normalization
    - Grouped-Query Attention (GQA) with RoPE
    - SwiGLU Feed-Forward Network
    - Residual skip connections
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads=num_kv_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_dim=ffn_hidden_dim)
        
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rope_emb: Optional[object] = None,
        rope_offset: int = 0
    ) -> np.ndarray:
        """
        Pre-LN Forward:
            h = x + Attn(RMSNorm(x))
            out = h + SwiGLU(RMSNorm(h))
        """
        self.x_input = x
        norm1_out = self.norm1.forward(x)
        attn_out = self.attn.forward(norm1_out, mask=mask, rope_emb=rope_emb, rope_offset=rope_offset)
        self.h = x + attn_out
        
        norm2_out = self.norm2.forward(self.h)
        ffn_out = self.ffn.forward(norm2_out)
        out = self.h + ffn_out
        return out
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through Pre-LN residual block."""
        # Gradient through FFN branch
        grad_norm2 = self.ffn.backward(grad_output)
        grad_h = grad_output + self.norm2.backward(grad_norm2)
        
        # Gradient through Attn branch
        grad_norm1 = self.attn.backward(grad_h)
        grad_x = grad_h + self.norm1.backward(grad_norm1)
        return grad_x

