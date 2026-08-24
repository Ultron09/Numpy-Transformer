"""
Low-Rank Adaptation (LoRA) for Parameter-Efficient Fine-Tuning (PEFT)

Implements LoRA (Hu et al., 2021) from first principles using pure NumPy.
Allows efficient adaptation of pre-trained transformer layers by freezing the
base model parameters and injecting trainable low-rank decomposition matrices.

Mathematical Formulation:
    h = W_0 x + ΔW x = W_0 x + (α / r) * B A x
    where:
        W_0 ∈ ℝ^{d_out × d_in} (Frozen pre-trained weights)
        A ∈ ℝ^{r × d_in} (Initialized with N(0, 1/r))
        B ∈ ℝ^{d_out × r} (Initialized with 0, ensuring ΔW = 0 at start)
        r ≪ min(d_in, d_out) (Rank)
        α (Constant scaling hyperparameter)
"""

from typing import Tuple, Optional, Dict, List
import numpy as np


class LoRALinear:
    """
    Linear layer with Low-Rank Adaptation (LoRA) adapters.
    
    Supports training low-rank parameters (A and B) while keeping the original weight W_0 frozen.
    Features seamless weight merging for zero-overhead production inference.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: float = 8.0,
        use_bias: bool = True,
    ):
        """
        Args:
            in_features: Input dimension (d_in)
            out_features: Output dimension (d_out)
            r: Rank of the low-rank decomposition (r << min(d_in, d_out))
            lora_alpha: Scaling factor (scaling = lora_alpha / r)
            use_bias: Whether to include additive bias
        """
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.use_bias = use_bias
        self.merged = False
        
        # Frozen base pre-trained weights W_0: shape (in_features, out_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * (1.0 / np.sqrt(in_features))
        self.bias = np.zeros(out_features, dtype=np.float32) if use_bias else None
        
        # Trainable LoRA adapter matrices
        if self.r > 0:
            # Matrix A: shape (in_features, r) initialized with Gaussian
            self.lora_A = np.random.randn(in_features, r).astype(np.float32) * (1.0 / np.sqrt(r))
            # Matrix B: shape (r, out_features) initialized to 0 (so Delta W starts at 0)
            self.lora_B = np.zeros((r, out_features), dtype=np.float32)
        else:
            self.lora_A = None
            self.lora_B = None
            
        # Gradients
        self.grad_lora_A = None
        self.grad_lora_B = None
        self.grad_bias = None
        
        # Cache for backpropagation
        self.cached_x = None
        self.cached_lora_A_out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: h = x W + bias + scaling * (x A) B
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        self.cached_x = x
        
        if self.merged or self.r == 0:
            out = np.matmul(x, self.W)
        else:
            # Base linear projection
            base_out = np.matmul(x, self.W)
            
            # Low-rank adapter branch: (x @ A) @ B
            lora_A_out = np.matmul(x, self.lora_A)  # shape (..., r)
            self.cached_lora_A_out = lora_A_out
            lora_out = np.matmul(lora_A_out, self.lora_B) * self.scaling  # shape (..., out_features)
            
            out = base_out + lora_out
            
        if self.bias is not None:
            out = out + self.bias
            
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradients w.r.t LoRA parameters and input x.
        Base weights W remain frozen (no gradient computed for W).
        
        Args:
            grad_output: Gradient from subsequent layer, shape (..., out_features)
            
        Returns:
            grad_x: Gradient with respect to input, shape (..., in_features)
        """
        if self.merged:
            raise RuntimeError("Cannot perform backward pass when weights are merged. Call unmerge_weights() first.")
            
        x = self.cached_x
        # Reshape to 2D for clean matrix calculus if higher dimension
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        grad_2d = grad_output.reshape(-1, self.out_features)
        
        # Bias gradient
        if self.bias is not None:
            self.grad_bias = np.sum(grad_2d, axis=0)
            
        # Base weight backprop to input
        grad_x_2d = np.matmul(grad_2d, self.W.T)
        
        if self.r > 0:
            # Gradient for LoRA B: dL/dB = scaling * (lora_A_out)^T @ grad_output
            lora_A_out_2d = self.cached_lora_A_out.reshape(-1, self.r)
            self.grad_lora_B = self.scaling * np.matmul(lora_A_out_2d.T, grad_2d)
            
            # Gradient for LoRA A: dL/dA = scaling * x^T @ (grad_output @ B^T)
            grad_lora_A_out = np.matmul(grad_2d, self.lora_B.T) * self.scaling
            self.grad_lora_A = np.matmul(x_2d.T, grad_lora_A_out)
            
            # Gradient w.r.t input through LoRA branch
            grad_x_lora = np.matmul(grad_lora_A_out, self.lora_A.T)
            grad_x_2d = grad_x_2d + grad_x_lora
            
        grad_x = grad_x_2d.reshape(orig_shape)
        return grad_x

    def merge_weights(self):
        """
        Merge low-rank adapter delta into the base weight matrix:
            W = W + scaling * (A @ B)
        Eliminates runtime inference latency.
        """
        if not self.merged and self.r > 0:
            delta_W = self.scaling * np.matmul(self.lora_A, self.lora_B)
            self.W = self.W + delta_W
            self.merged = True

    def unmerge_weights(self):
        """
        Unmerge low-rank adapter delta from base weights to resume fine-tuning.
        """
        if self.merged and self.r > 0:
            delta_W = self.scaling * np.matmul(self.lora_A, self.lora_B)
            self.W = self.W - delta_W
            self.merged = False

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        """Return dictionary of trainable adapter parameters."""
        params = {}
        if self.r > 0:
            params['lora_A'] = self.lora_A
            params['lora_B'] = self.lora_B
        if self.bias is not None:
            params['bias'] = self.bias
        return params

    def get_trainable_grads(self) -> Dict[str, np.ndarray]:
        """Return dictionary of gradients for trainable parameters."""
        grads = {}
        if self.r > 0:
            grads['lora_A'] = self.grad_lora_A
            grads['lora_B'] = self.grad_lora_B
        if self.bias is not None:
            grads['bias'] = self.grad_bias
        return grads

    @classmethod
    def from_linear(cls, linear_layer, r: int = 4, lora_alpha: float = 8.0) -> 'LoRALinear':
        """
        Create a LoRALinear layer initialized with weights from an existing Linear layer.
        """
        in_f = linear_layer.W.shape[0]
        out_f = linear_layer.W.shape[1]
        use_bias = hasattr(linear_layer, 'b') and linear_layer.b is not None
        
        lora_layer = cls(in_features=in_f, out_features=out_f, r=r, lora_alpha=lora_alpha, use_bias=use_bias)
        lora_layer.W = linear_layer.W.copy()
        if use_bias:
            lora_layer.bias = linear_layer.b.copy() if hasattr(linear_layer, 'b') else linear_layer.bias.copy()
        return lora_layer
