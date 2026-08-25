"""
Sparse Mixture of Experts (MoE) Feed-Forward Network Layer

Implements sparse conditional computation (Shazeer et al., 2017; Fedus et al., 2022; Mixtral, 2024).
Routes individual tokens to the top-k most relevant expert networks, vastly increasing
model capacity without proportional compute overhead.

Key Features:
- Noisy Top-K Gating router with learnable exploration noise
- Switch / Mixtral style Top-1 and Top-2 expert dispatch and aggregation
- Auxiliary load-balancing loss calculation to prevent expert collapse
- Pure NumPy forward and backward passes
"""

from typing import List, Tuple, Optional, Dict
import numpy as np


class ExpertFFN:
    """Single Feed-Forward Expert block using GELU activation."""
    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        # Weights: W1: (d_model, d_ff), W2: (d_ff, d_model)
        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * (1.0 / np.sqrt(d_model))
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * (1.0 / np.sqrt(d_ff))
        self.b2 = np.zeros(d_model, dtype=np.float32)
        
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        
        self.cached_x = None
        self.cached_h = None
        self.cached_gelu = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: shape (N, d_model)
        self.cached_x = x
        h = np.matmul(x, self.W1) + self.b1
        self.cached_h = h
        
        # GELU activation: 0.5 * h * (1 + tanh(sqrt(2/pi) * (h + 0.044715 * h^3)))
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        tanh_val = np.tanh(sqrt_2_pi * (h + 0.044715 * (h ** 3)))
        gelu_out = 0.5 * h * (1.0 + tanh_val)
        self.cached_gelu = gelu_out
        
        out = np.matmul(gelu_out, self.W2) + self.b2
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_output: shape (N, d_model)
        self.grad_W2 = np.matmul(self.cached_gelu.T, grad_output)
        self.grad_b2 = np.sum(grad_output, axis=0)
        
        grad_gelu = np.matmul(grad_output, self.W2.T)
        
        # GELU derivative
        h = self.cached_h
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        tanh_val = np.tanh(sqrt_2_pi * (h + 0.044715 * (h ** 3)))
        cdf = 0.5 * (1.0 + tanh_val)
        d_tanh = sqrt_2_pi * (1.0 + 0.044715 * 3.0 * (h ** 2))
        sech2 = 1.0 - tanh_val ** 2
        d_gelu = cdf + 0.5 * h * sech2 * d_tanh
        
        grad_h = grad_gelu * d_gelu
        self.grad_W1 = np.matmul(self.cached_x.T, grad_h)
        self.grad_b1 = np.sum(grad_h, axis=0)
        
        grad_x = np.matmul(grad_h, self.W1.T)
        return grad_x


class NoisyTopKGating:
    """
    Learnable router network that computes gating weights and routes tokens to top-k experts.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, noise_epsilon: float = 1e-2):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_epsilon = noise_epsilon
        
        # Gating projection matrix: W_gate ∈ ℝ^{d_model × num_experts}
        self.W_gate = np.random.randn(d_model, num_experts).astype(np.float32) * (1.0 / np.sqrt(d_model))
        # Noise weight matrix: W_noise ∈ ℝ^{d_model × num_experts}
        self.W_noise = np.random.randn(d_model, num_experts).astype(np.float32) * (1.0 / np.sqrt(d_model))
        
        self.grad_W_gate = None
        self.grad_W_noise = None

    def forward(
        self,
        x: np.ndarray,
        is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute routing weights and auxiliary load balancing loss.
        
        Args:
            x: Input tensor of shape (N, d_model)
            is_training: Whether to add exploration noise
            
        Returns:
            top_k_weights: shape (N, top_k) normalized softmax weights
            top_k_indices: shape (N, top_k) integer indices of selected experts
            full_router_probs: shape (N, num_experts) full softmax distribution
            aux_loss: Scalar auxiliary load balancing loss
        """
        N = x.shape[0]
        gate_logits = np.matmul(x, self.W_gate)  # (N, num_experts)
        
        if is_training and self.noise_epsilon > 0:
            # Add learnable noise: standard_normal * softplus(x @ W_noise)
            noise_std = np.log1p(np.exp(np.clip(np.matmul(x, self.W_noise), -20, 20)))
            noise = np.random.randn(*gate_logits.shape).astype(np.float32) * noise_std
            routing_logits = gate_logits + noise
        else:
            routing_logits = gate_logits
            
        # Top-k selection per token
        # Get top-k indices
        top_k_indices = np.zeros((N, self.top_k), dtype=np.int32)
        top_k_weights = np.zeros((N, self.top_k), dtype=np.float32)
        
        for i in range(N):
            token_logits = routing_logits[i]
            indices = np.argpartition(token_logits, -self.top_k)[-self.top_k:]
            sorted_indices = indices[np.argsort(token_logits[indices])[::-1]]
            top_k_indices[i] = sorted_indices
            
            # Softmax over top-k logits
            selected_logits = token_logits[sorted_indices]
            exp_logits = np.exp(selected_logits - np.max(selected_logits))
            top_k_weights[i] = exp_logits / (np.sum(exp_logits) + 1e-12)
            
        # Full softmax probability across all experts (for auxiliary loss)
        exp_full = np.exp(gate_logits - np.max(gate_logits, axis=-1, keepdims=True))
        full_router_probs = exp_full / (np.sum(exp_full, axis=-1, keepdims=True) + 1e-12)
        
        # Auxiliary load balancing loss (Switch / GShard formula):
        # L_aux = num_experts * sum(fraction_routed * average_probability)
        # Fraction of tokens routed to each expert
        expert_mask = np.zeros((N, self.num_experts), dtype=np.float32)
        for i in range(N):
            expert_mask[i, top_k_indices[i]] = 1.0
        fraction_routed = np.mean(expert_mask, axis=0)  # shape (num_experts,)
        avg_prob = np.mean(full_router_probs, axis=0)   # shape (num_experts,)
        aux_loss = float(self.num_experts * np.sum(fraction_routed * avg_prob))
        
        return top_k_weights, top_k_indices, full_router_probs, aux_loss


class SparseMoEFFN:
    """
    Sparse Mixture of Experts Feed-Forward Layer.
    
    Dynamically routes each token in the batch through its designated top-k experts
    and computes the weighted combination of expert outputs.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coeff = aux_loss_coeff
        
        self.router = NoisyTopKGating(d_model=d_model, num_experts=num_experts, top_k=top_k)
        self.experts = [ExpertFFN(d_model=d_model, d_ff=d_ff) for _ in range(num_experts)]
        
        self.last_aux_loss = 0.0
        self.cached_x = None
        self.cached_top_k_weights = None
        self.cached_top_k_indices = None
        self.cached_orig_shape = None

    def forward(self, x: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Forward pass through Sparse MoE FFN layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (N, d_model)
            is_training: Whether in training mode
            
        Returns:
            Output tensor of shape identical to x
        """
        self.cached_orig_shape = x.shape
        x_2d = x.reshape(-1, self.d_model)
        N = x_2d.shape[0]
        self.cached_x = x_2d
        
        # Route tokens
        top_k_weights, top_k_indices, _, aux_loss = self.router.forward(x_2d, is_training=is_training)
        self.last_aux_loss = aux_loss * self.aux_loss_coeff
        self.cached_top_k_weights = top_k_weights
        self.cached_top_k_indices = top_k_indices
        
        output = np.zeros_like(x_2d, dtype=np.float32)
        
        # Dispatch and execute per expert
        for expert_id in range(self.num_experts):
            # Find tokens routed to this expert
            token_mask = np.any(top_k_indices == expert_id, axis=-1)
            if not np.any(token_mask):
                continue
                
            token_idx = np.where(token_mask)[0]
            expert_in = x_2d[token_idx]
            expert_out = self.experts[expert_id].forward(expert_in)
            
            # Combine weighted outputs
            for i, tid in enumerate(token_idx):
                k_pos = np.where(top_k_indices[tid] == expert_id)[0][0]
                weight = top_k_weights[tid, k_pos]
                output[tid] += weight * expert_out[i]
                
        return output.reshape(self.cached_orig_shape)
