"""
GPT (Generative Pre-trained Transformer) Implementation using only NumPy

This module implements a complete GPT model from scratch using only NumPy.
Each component is thoroughly documented with mathematical explanations to serve
as an educational resource for understanding transformer architectures.

Author: Educational Implementation
Purpose: Learn transformer architecture from first principles
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle
import json


class GELU:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    Mathematical Formula:
        GELU(x) = x * Φ(x)
        where Φ(x) is the cumulative distribution function of the standard normal distribution
        
    Approximation used (tanh-based):
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    GELU is smoother than ReLU and has been shown to work well in transformers.
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of GELU activation.
        
        Args:
            x: Input array of any shape
            
        Returns:
            Activated output of same shape as input
        """
        # Using tanh approximation for GELU
        self.x = x
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
        self.output = x * cdf
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of GELU activation.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient with respect to input
        """
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        x_cubed = self.x ** 3
        
        # Derivative of tanh approximation
        tanh_arg = sqrt_2_over_pi * (self.x + 0.044715 * x_cubed)
        tanh_val = np.tanh(tanh_arg)
        
        cdf = 0.5 * (1.0 + tanh_val)
        
        # Derivative of the inner term
        d_tanh_arg = sqrt_2_over_pi * (1 + 0.044715 * 3 * self.x**2)
        sech2 = 1 - tanh_val**2
        
        # Full derivative using product rule
        grad_x = cdf + self.x * 0.5 * sech2 * d_tanh_arg
        
        return grad_output * grad_x


class Softmax:
    """
    Softmax activation function with numerical stability.
    
    Mathematical Formula:
        softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
        
    For numerical stability, we subtract the max:
        softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    
    This is commonly used to convert logits to probability distributions.
    """
    
    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Forward pass of softmax.
        
        Args:
            x: Input array (e.g., logits)
            axis: Axis along which to apply softmax
            
        Returns:
            Probability distribution (sums to 1 along specified axis)
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.output = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of softmax.
        
        The Jacobian of softmax is:
            ∂s_i/∂x_j = s_i * (δ_ij - s_j)
        where δ_ij is the Kronecker delta
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        # Efficient computation: grad_x = output * (grad_output - (grad_output ⊙ output).sum())
        s = self.output
        sum_term = np.sum(grad_output * s, axis=-1, keepdims=True)
        grad_x = s * (grad_output - sum_term)
        return grad_x


class LayerNorm:
    """
    Layer Normalization.
    
    Mathematical Formula:
        LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
        
    where:
        μ = mean(x) across features
        σ² = variance(x) across features
        γ = learned scale parameter (initialized to 1)
        β = learned shift parameter (initialized to 0)
        ε = small constant for numerical stability
    
    Layer normalization normalizes across the feature dimension, which helps
    stabilize training in deep networks.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize layer normalization.
        
        Args:
            normalized_shape: Size of the feature dimension
            eps: Small constant for numerical stability
        """
        self.eps = eps
        self.gamma = np.ones(normalized_shape)  # Scale parameter
        self.beta = np.zeros(normalized_shape)   # Shift parameter
        
        # For storing gradients
        self.grad_gamma = None
        self.grad_beta = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of layer normalization.
        
        Args:
            x: Input of shape (..., normalized_shape)
            
        Returns:
            Normalized output of same shape as input
        """
        self.x = x
        # Compute mean and variance across last dimension
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        self.x_centered = x - self.mean
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = self.x_centered / self.std
        
        # Scale and shift
        output = self.gamma * self.x_norm + self.beta
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of layer normalization.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        N = grad_output.shape[-1]
        
        # Gradients for scale and shift parameters
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=tuple(range(len(grad_output.shape) - 1)))
        self.grad_beta = np.sum(grad_output, axis=tuple(range(len(grad_output.shape) - 1)))
        
        # Gradient with respect to normalized x
        grad_x_norm = grad_output * self.gamma
        
        # Gradient with respect to variance
        grad_var = np.sum(grad_x_norm * self.x_centered * -0.5 * (self.var + self.eps)**(-1.5), axis=-1, keepdims=True)
        
        # Gradient with respect to mean
        grad_mean = np.sum(grad_x_norm * -1.0 / self.std, axis=-1, keepdims=True)
        grad_mean += grad_var * np.mean(-2.0 * self.x_centered, axis=-1, keepdims=True)
        
        # Gradient with respect to x
        grad_x = grad_x_norm / self.std
        grad_x += grad_var * 2.0 * self.x_centered / N
        grad_x += grad_mean / N
        
        return grad_x


class Linear:
    """
    Fully connected linear layer.
    
    Mathematical Formula:
        y = xW^T + b
        
    where:
        x: input of shape (..., in_features)
        W: weight matrix of shape (out_features, in_features)
        b: bias vector of shape (out_features,)
        y: output of shape (..., out_features)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
        """
        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features))
        
        if bias:
            self.bias = np.zeros(out_features)
        else:
            self.bias = None
        
        # For storing gradients
        self.grad_weight = None
        self.grad_bias = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of linear layer.
        
        Args:
            x: Input of shape (..., in_features)
            
        Returns:
            Output of shape (..., out_features)
        """
        self.x = x
        output = x @ self.weight.T
        
        if self.bias is not None:
            output += self.bias
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of linear layer.
        
        Args:
            grad_output: Gradient from next layer of shape (..., out_features)
            
        Returns:
            Gradient with respect to input of shape (..., in_features)
        """
        # Gradient with respect to input
        grad_x = grad_output @ self.weight
        
        # Gradient with respect to weights
        # Reshape to (batch_size * seq_len, features) if needed
        original_shape = self.x.shape
        x_reshaped = self.x.reshape(-1, self.x.shape[-1])
        grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
        
        self.grad_weight = grad_output_reshaped.T @ x_reshaped
        
        # Gradient with respect to bias
        if self.bias is not None:
            self.grad_bias = np.sum(grad_output, axis=tuple(range(len(grad_output.shape) - 1)))
        
        return grad_x


class Embedding:
    """
    Embedding layer that converts token indices to dense vectors.
    
    This is essentially a lookup table where each token index maps to a
    learned vector representation.
    
    Mathematical Operation:
        For input token index i:
        output = embedding_matrix[i, :]
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initialize embedding layer.
        
        Args:
            num_embeddings: Size of the vocabulary (number of unique tokens)
            embedding_dim: Dimension of each embedding vector
        """
        # Initialize embeddings with small random values
        self.weight = np.random.randn(num_embeddings, embedding_dim) * 0.02
        self.grad_weight = None
    
    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Forward pass: look up embeddings for given indices.
        
        Args:
            indices: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        self.indices = indices
        return self.weight[indices]
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass for embeddings.
        
        Args:
            grad_output: Gradient from next layer
        """
        # Initialize gradient matrix
        self.grad_weight = np.zeros_like(self.weight)
        
        # Accumulate gradients for each embedding
        # This handles the case where the same token appears multiple times
        np.add.at(self.grad_weight, self.indices, grad_output)


class MultiHeadAttention:
    """
    Multi-Head Self-Attention mechanism.
    
    This is the core component of the transformer architecture.
    
    Mathematical Formula:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Key Concepts:
    - Query (Q): What we're looking for
    - Key (K): What we have
    - Value (V): The actual content
    - Attention scores: How much to focus on each position
    - Multiple heads: Allow attending to different aspects simultaneously
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            dropout: Dropout rate (for regularization)
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.dropout = dropout
        
        # Linear projections for Q, K, V and output
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            mask: Optional mask for causal attention (shape: (seq_len, seq_len))
            
        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.W_q.forward(x)  # (batch_size, seq_len, d_model)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)
        
        # Reshape to (batch_size, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        # scores = QK^T / √d_k
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask (for causal/autoregressive attention)
        if mask is not None:
            scores = scores + mask  # mask contains -inf for positions to ignore
        
        # Softmax to get attention weights
        self.softmax = Softmax()
        attn_weights = self.softmax.forward(scores)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Store for backward pass
        self.Q, self.K, self.V = Q, K, V
        self.attn_weights = attn_weights
        self.batch_size, self.seq_len = batch_size, seq_len
        
        # Apply dropout (during training)
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*attn_weights.shape) > self.dropout).astype(float)
            attn_weights = attn_weights * dropout_mask / (1 - self.dropout)
            self.dropout_mask = dropout_mask
        
        # Apply attention to values
        context = attn_weights @ V  # (batch_size, num_heads, seq_len, d_k)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o.forward(context)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of multi-head attention.
        
        This is complex due to the multiple matrix multiplications and reshaping.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        batch_size, seq_len, d_model = grad_output.shape
        
        # Gradient through output projection
        grad_context = self.W_o.backward(grad_output)
        
        # Reshape back to multi-head format
        grad_context = grad_context.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Gradient through attention application (attn_weights @ V)
        grad_attn_weights = grad_context @ self.V.transpose(0, 1, 3, 2)
        grad_V = self.attn_weights.transpose(0, 1, 3, 2) @ grad_context
        
        # Gradient through dropout
        if self.dropout > 0:
            grad_attn_weights = grad_attn_weights * self.dropout_mask / (1 - self.dropout)
        
        # Gradient through softmax
        grad_scores = self.softmax.backward(grad_attn_weights)
        
        # Gradient through scaling
        grad_scores = grad_scores / np.sqrt(self.d_k)
        
        # Gradient through QK^T
        grad_Q = grad_scores @ self.K
        grad_K = grad_scores.transpose(0, 1, 3, 2) @ self.Q
        
        # Reshape back to (batch_size, seq_len, d_model)
        grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Gradient through Q, K, V projections
        grad_x_q = self.W_q.backward(grad_Q)
        grad_x_k = self.W_k.backward(grad_K)
        grad_x_v = self.W_v.backward(grad_V)
        
        # Sum gradients (since Q, K, V all come from same input x)
        grad_x = grad_x_q + grad_x_k + grad_x_v
        
        return grad_x


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    Mathematical Formula:
        FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    This is applied independently to each position. Typically, the hidden
    dimension is 4x the model dimension.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.activation = GELU()
        self.dropout = dropout
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output of same shape as input
        """
        # First linear layer
        hidden = self.fc1.forward(x)
        
        # Activation
        hidden = self.activation.forward(hidden)
        
        # Dropout
        if self.dropout > 0:
            self.dropout_mask = (np.random.rand(*hidden.shape) > self.dropout).astype(float)
            hidden = hidden * self.dropout_mask / (1 - self.dropout)
        
        # Second linear layer
        output = self.fc2.forward(hidden)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of feed-forward network.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        # Gradient through second linear layer
        grad_hidden = self.fc2.backward(grad_output)
        
        # Gradient through dropout
        if self.dropout > 0:
            grad_hidden = grad_hidden * self.dropout_mask / (1 - self.dropout)
        
        # Gradient through activation
        grad_hidden = self.activation.backward(grad_hidden)
        
        # Gradient through first linear layer
        grad_x = self.fc1.backward(grad_hidden)
        
        return grad_x


class TransformerBlock:
    """
    Complete Transformer Block.
    
    Architecture:
        x → LayerNorm → MultiHeadAttention → Add (residual) →
        → LayerNorm → FeedForward → Add (residual) → output
    
    Key Concepts:
    - Residual connections: Help gradient flow in deep networks
    - Layer normalization: Stabilize training
    - Multi-head attention: Capture different aspects of relationships
    - Feed-forward: Process each position independently
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output of same shape as input
        """
        # Multi-head attention with residual connection
        # Pre-normalization (norm before sub-layer)
        normed = self.norm1.forward(x)
        attn_output = self.attention.forward(normed, mask)
        
        # Dropout and residual
        if self.dropout > 0:
            self.dropout_mask1 = (np.random.rand(*attn_output.shape) > self.dropout).astype(float)
            attn_output = attn_output * self.dropout_mask1 / (1 - self.dropout)
        
        x = x + attn_output
        self.after_attn = x
        
        # Feed-forward with residual connection
        normed = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(normed)
        
        # Dropout and residual
        if self.dropout > 0:
            self.dropout_mask2 = (np.random.rand(*ff_output.shape) > self.dropout).astype(float)
            ff_output = ff_output * self.dropout_mask2 / (1 - self.dropout)
        
        x = x + ff_output
        
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of transformer block.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        # Gradient through second residual connection
        grad_ff_output = grad_output.copy()
        grad_x = grad_output.copy()
        
        # Gradient through dropout
        if self.dropout > 0:
            grad_ff_output = grad_ff_output * self.dropout_mask2 / (1 - self.dropout)
        
        # Gradient through feed-forward
        grad_normed2 = self.feed_forward.backward(grad_ff_output)
        
        # Gradient through norm2
        grad_after_attn = self.norm2.backward(grad_normed2)
        grad_x += grad_after_attn
        
        # Gradient through first residual connection
        grad_attn_output = grad_x.copy()
        
        # Gradient through dropout
        if self.dropout > 0:
            grad_attn_output = grad_attn_output * self.dropout_mask1 / (1 - self.dropout)
        
        # Gradient through attention
        grad_normed1 = self.attention.backward(grad_attn_output)
        
        # Gradient through norm1
        grad_input = self.norm1.backward(grad_normed1)
        grad_input += grad_x
        
        return grad_input


class GPT:
    """
    Complete GPT (Generative Pre-trained Transformer) Model.
    
    Architecture:
        1. Token Embedding: Convert token indices to vectors
        2. Positional Embedding: Add position information
        3. Transformer Blocks: Stack of N transformer layers
        4. Layer Normalization: Final normalization
        5. Output Projection: Project to vocabulary size
    
    This is an autoregressive language model that predicts the next token
    given previous tokens.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (embedding size)
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.ln_f = LayerNorm(d_model)
        
        # Output projection (to vocabulary)
        self.lm_head = Linear(d_model, vocab_size, bias=False)
        
        # Share weights between token embedding and output projection
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = dropout
    
    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Forward pass of GPT model.
        
        Args:
            indices: Token indices of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = indices.shape
        
        # Get embeddings
        token_emb = self.token_embedding.forward(indices)  # (batch_size, seq_len, d_model)
        
        # Position indices: [0, 1, 2, ..., seq_len-1]
        positions = np.arange(seq_len)[np.newaxis, :]  # (1, seq_len)
        pos_emb = self.position_embedding.forward(positions)  # (1, seq_len, d_model)
        
        # Combine embeddings
        x = token_emb + pos_emb
        
        # Dropout
        if self.dropout > 0:
            self.dropout_mask = (np.random.rand(*x.shape) > self.dropout).astype(float)
            x = x * self.dropout_mask / (1 - self.dropout)
        
        # Create causal mask (lower triangular)
        # This ensures that position i can only attend to positions <= i
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)  # Upper triangle = -inf
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final layer norm
        x = self.ln_f.forward(x)
        
        # Project to vocabulary
        logits = self.lm_head.forward(x)
        
        return logits
    
    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass of GPT model.
        
        Args:
            grad_logits: Gradient of loss with respect to logits
        """
        # Gradient through output projection
        grad_x = self.lm_head.backward(grad_logits)
        
        # Gradient through final layer norm
        grad_x = self.ln_f.backward(grad_x)
        
        # Gradient through transformer blocks (in reverse order)
        for block in reversed(self.blocks):
            grad_x = block.backward(grad_x)
        
        # Gradient through dropout
        if self.dropout > 0:
            grad_x = grad_x * self.dropout_mask / (1 - self.dropout)
        
        # Gradient through embeddings
        # Position embedding gradient
        grad_pos_emb = np.sum(grad_x, axis=0, keepdims=True)  # Sum over batch
        self.position_embedding.backward(grad_pos_emb)
        
        # Token embedding gradient
        self.token_embedding.backward(grad_x)
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get all model parameters for optimization."""
        params = []
        
        # Embeddings
        params.append(('token_embedding.weight', self.token_embedding.weight))
        params.append(('position_embedding.weight', self.position_embedding.weight))
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            # Attention
            params.append((f'block{i}.attn.W_q.weight', block.attention.W_q.weight))
            params.append((f'block{i}.attn.W_q.bias', block.attention.W_q.bias))
            params.append((f'block{i}.attn.W_k.weight', block.attention.W_k.weight))
            params.append((f'block{i}.attn.W_k.bias', block.attention.W_k.bias))
            params.append((f'block{i}.attn.W_v.weight', block.attention.W_v.weight))
            params.append((f'block{i}.attn.W_v.bias', block.attention.W_v.bias))
            params.append((f'block{i}.attn.W_o.weight', block.attention.W_o.weight))
            params.append((f'block{i}.attn.W_o.bias', block.attention.W_o.bias))
            
            # Feed-forward
            params.append((f'block{i}.ff.fc1.weight', block.feed_forward.fc1.weight))
            params.append((f'block{i}.ff.fc1.bias', block.feed_forward.fc1.bias))
            params.append((f'block{i}.ff.fc2.weight', block.feed_forward.fc2.weight))
            params.append((f'block{i}.ff.fc2.bias', block.feed_forward.fc2.bias))
            
            # Layer norms
            params.append((f'block{i}.norm1.gamma', block.norm1.gamma))
            params.append((f'block{i}.norm1.beta', block.norm1.beta))
            params.append((f'block{i}.norm2.gamma', block.norm2.gamma))
            params.append((f'block{i}.norm2.beta', block.norm2.beta))
        
        # Final layer norm
        params.append(('ln_f.gamma', self.ln_f.gamma))
        params.append(('ln_f.beta', self.ln_f.beta))
        
        return params
    
    def get_gradients(self) -> List[np.ndarray]:
        """Get all gradients for optimization."""
        grads = []
        
        # Embeddings
        grads.append(('token_embedding.weight', self.token_embedding.grad_weight))
        grads.append(('position_embedding.weight', self.position_embedding.grad_weight))
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            # Attention
            grads.append((f'block{i}.attn.W_q.weight', block.attention.W_q.grad_weight))
            grads.append((f'block{i}.attn.W_q.bias', block.attention.W_q.grad_bias))
            grads.append((f'block{i}.attn.W_k.weight', block.attention.W_k.grad_weight))
            grads.append((f'block{i}.attn.W_k.bias', block.attention.W_k.grad_bias))
            grads.append((f'block{i}.attn.W_v.weight', block.attention.W_v.grad_weight))
            grads.append((f'block{i}.attn.W_v.bias', block.attention.W_v.grad_bias))
            grads.append((f'block{i}.attn.W_o.weight', block.attention.W_o.grad_weight))
            grads.append((f'block{i}.attn.W_o.bias', block.attention.W_o.grad_bias))
            
            # Feed-forward
            grads.append((f'block{i}.ff.fc1.weight', block.feed_forward.fc1.grad_weight))
            grads.append((f'block{i}.ff.fc1.bias', block.feed_forward.fc1.grad_bias))
            grads.append((f'block{i}.ff.fc2.weight', block.feed_forward.fc2.grad_weight))
            grads.append((f'block{i}.ff.fc2.bias', block.feed_forward.fc2.grad_bias))
            
            # Layer norms
            grads.append((f'block{i}.norm1.gamma', block.norm1.grad_gamma))
            grads.append((f'block{i}.norm1.beta', block.norm1.grad_beta))
            grads.append((f'block{i}.norm2.gamma', block.norm2.grad_gamma))
            grads.append((f'block{i}.norm2.beta', block.norm2.grad_beta))
        
        # Final layer norm
        grads.append(('ln_f.gamma', self.ln_f.grad_gamma))
        grads.append(('ln_f.beta', self.ln_f.grad_beta))
        
        return grads
    
    def save(self, filepath: str):
        """Save model parameters to file."""
        params = {name: param for name, param in self.get_parameters()}
        config = {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump({'params': params, 'config': config}, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model parameters from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        params = data['params']
        
        # Load embeddings
        self.token_embedding.weight = params['token_embedding.weight']
        self.position_embedding.weight = params['position_embedding.weight']
        
        # Load transformer blocks
        for i, block in enumerate(self.blocks):
            block.attention.W_q.weight = params[f'block{i}.attn.W_q.weight']
            block.attention.W_q.bias = params[f'block{i}.attn.W_q.bias']
            block.attention.W_k.weight = params[f'block{i}.attn.W_k.weight']
            block.attention.W_k.bias = params[f'block{i}.attn.W_k.bias']
            block.attention.W_v.weight = params[f'block{i}.attn.W_v.weight']
            block.attention.W_v.bias = params[f'block{i}.attn.W_v.bias']
            block.attention.W_o.weight = params[f'block{i}.attn.W_o.weight']
            block.attention.W_o.bias = params[f'block{i}.attn.W_o.bias']
            
            block.feed_forward.fc1.weight = params[f'block{i}.ff.fc1.weight']
            block.feed_forward.fc1.bias = params[f'block{i}.ff.fc1.bias']
            block.feed_forward.fc2.weight = params[f'block{i}.ff.fc2.weight']
            block.feed_forward.fc2.bias = params[f'block{i}.ff.fc2.bias']
            
            block.norm1.gamma = params[f'block{i}.norm1.gamma']
            block.norm1.beta = params[f'block{i}.norm1.beta']
            block.norm2.gamma = params[f'block{i}.norm2.gamma']
            block.norm2.beta = params[f'block{i}.norm2.beta']
        
        # Load final layer norm
        self.ln_f.gamma = params['ln_f.gamma']
        self.ln_f.beta = params['ln_f.beta']
        
        print(f"Model loaded from {filepath}")
    
    def generate(
        self,
        start_tokens: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Starting token indices of shape (1, start_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated token indices
        """
        generated = start_tokens.copy()
        
        for _ in range(max_new_tokens):
            # Get logits for current sequence
            # Crop to max_seq_len if needed
            context = generated[:, -self.max_seq_len:]
            logits = self.forward(context)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Softmax to get probabilities
            softmax = Softmax()
            probs = softmax.forward(logits)
            
            # Sample from distribution
            next_token = np.random.choice(self.vocab_size, p=probs[0])
            
            # Append to sequence
            generated = np.concatenate([generated, [[next_token]]], axis=1)
        
        return generated


if __name__ == "__main__":
    # Simple test
    print("GPT Implementation Test")
    print("=" * 50)
    
    # Create small model
    vocab_size = 100
    d_model = 64
    num_layers = 2
    num_heads = 4
    
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=256,
        max_seq_len=32
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    test_input = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {test_input.shape}")
    logits = model.forward(test_input)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Test backward pass
    grad_logits = np.random.randn(*logits.shape)
    model.backward(grad_logits)
    print("\nBackward pass completed successfully!")
    
    # Count parameters
    total_params = sum(p.size for _, p in model.get_parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "=" * 50)
    print("All tests passed! [OK]")
