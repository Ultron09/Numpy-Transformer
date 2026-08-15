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
