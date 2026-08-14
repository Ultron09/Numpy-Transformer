"""
Optimizers & Learning Rate Schedulers for Transformer Training

Includes:
- AdamW: Decoupled Weight Decay Optimizer (Loshchilov & Hutter)
- SGDMomentum: Stochastic Gradient Descent with Nesterov accelerated momentum
- CosineAnnealingWarmupLR: Cosine learning rate decay with linear warmup
- LinearWarmupDecayLR: Linear warmup followed by linear decay
- clip_grad_norm: Global L2 gradient norm clipping
"""

import math
from typing import List, Tuple, Dict, Optional, Union
import numpy as np


def clip_grad_norm(
    grads: List[Tuple[str, np.ndarray]],
    max_norm: float = 1.0,
    norm_type: float = 2.0
) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    
    Args:
        grads: List of (name, gradient array) tuples
        max_norm: Max norm of the gradients
        norm_type: Type of the used p-norm (default: 2.0 for L2 norm)
        
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    valid_grads = [g for _, g in grads if g is not None]
    if len(valid_grads) == 0:
        return 0.0
        
    if norm_type == float('inf'):
        total_norm = max(np.max(np.abs(g)) for g in valid_grads)
    else:
        total_norm = 0.0
        for g in valid_grads:
            param_norm = np.sum(np.abs(g) ** norm_type)
            total_norm += param_norm
        total_norm = float(total_norm ** (1.0 / norm_type))
        
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for _, g in grads:
            if g is not None:
                g *= clip_coef
                
    return total_norm


class AdamW:
    """
    AdamW Optimizer with Decoupled Weight Decay (Fixing Weight Decay Regularization in Adam).
    
    Mathematical Formulation:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
    """
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        no_decay_params: Optional[List[str]] = None
    ):
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.no_decay_params = set(no_decay_params or ["bias", "gamma", "beta", "ln", "norm"])
        
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t: int = 0
        
    def step(self, params: List[Tuple[str, np.ndarray]], grads: List[Tuple[str, np.ndarray]]):
        """Perform a single optimization step."""
        self.t += 1
        
        # Bias corrections
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)
        step_size = self.lr / bias_correction1
        
        for (name, p), (g_name, g) in zip(params, grads):
            if g is None:
                continue
                
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
                
            # Update biased 1st & 2nd moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g ** 2)
            
            # Compute denom with bias-corrected 2nd moment
            denom = np.sqrt(self.v[name] / bias_correction2) + self.eps
            
            # Apply adaptive gradient update
            p_update = self.m[name] / denom
            
            # Apply decoupled weight decay (skip biases and normalization scales)
            skip_decay = any(nd in name.lower() for nd in self.no_decay_params)
            if self.weight_decay > 0 and not skip_decay:
                p -= self.lr * self.weight_decay * p
                
            p -= step_size * p_update


class SGDMomentum:
    """
    SGD with Momentum and Nesterov Acceleration.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: float = 0.0
    ):
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity: Dict[str, np.ndarray] = {}
        
    def step(self, params: List[Tuple[str, np.ndarray]], grads: List[Tuple[str, np.ndarray]]):
        for (name, p), (_, g) in zip(params, grads):
            if g is None:
                continue
                
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
                
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(p)
                
            v = self.momentum * self.velocity[name] + g
            self.velocity[name] = v
            
            if self.nesterov:
                update = self.momentum * v + g
            else:
                update = v
                
            p -= self.lr * update


class CosineAnnealingWarmupLR:
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup.
    
    Warmup phase:
        lr(t) = base_lr * (t / warmup_steps)
    Cosine decay phase:
        lr(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * (t - warmup) / (max_steps - warmup)))
    """
    
    def __init__(
        self,
        optimizer: Union[AdamW, SGDMomentum],
        base_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self) -> float:
        """Advance one step and update optimizer learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        self.optimizer.lr = lr
        return lr
        
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.base_lr * (float(self.current_step) / float(max(1, self.warmup_steps)))
        elif self.current_step > self.max_steps:
            return self.min_lr
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
