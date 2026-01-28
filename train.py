"""
Training Script for GPT Model

This script implements the complete training infrastructure including:
- Adam optimizer with momentum
- Cross-entropy loss
- Data loading and batching
- Training loop with validation
- Model checkpointing

Educational focus: Understanding how transformer models are trained.
"""

import numpy as np
from gpt_numpy import GPT
import time
import os
from typing import Tuple, List, Dict


class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) Optimizer.
    
    Adam combines the benefits of:
    - RMSprop: Uses moving average of squared gradients
    - Momentum: Uses moving average of gradients
    
    Mathematical Formulas:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          (First moment)
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         (Second moment)
        
        m̂_t = m_t / (1 - β₁^t)                       (Bias correction)
        v̂_t = v_t / (1 - β₂^t)                       (Bias correction)
        
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        (Parameter update)
    
    where:
        g_t: gradient at time t
        m_t: first moment (mean) of gradients
        v_t: second moment (uncentered variance) of gradients
        β₁, β₂: decay rates (typically 0.9 and 0.999)
        α: learning rate
        ε: small constant for numerical stability
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
            beta1: Decay rate for first moment (momentum)
            beta2: Decay rate for second moment (RMSprop)
            epsilon: Small constant for numerical stability
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Dictionary to store first and second moments for each parameter
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params: List[Tuple[str, np.ndarray]], grads: List[Tuple[str, np.ndarray]]):
        """
        Update parameters using Adam algorithm.
        
        Args:
            params: List of (name, parameter array) tuples
            grads: List of (name, gradient array) tuples
        """
        self.t += 1  # Increment time step
        
        for (param_name, param), (grad_name, grad) in zip(params, grads):
            assert param_name == grad_name, f"Parameter name mismatch: {param_name} vs {grad_name}"
            
            if grad is None:
                continue
            
            # Initialize moments if first time
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param)
                self.v[param_name] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss and its gradient.
    
    Mathematical Formula:
        Loss = -1/N * Σ log(softmax(logits)[target])
        
    where N is the total number of predictions.
    
    For numerical stability, we use the log-sum-exp trick:
        log(softmax(x_i)) = x_i - log(Σ exp(x_j))
                          = x_i - log_sum_exp(x)
    
    Args:
        logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token indices of shape (batch_size, seq_len)
        
    Returns:
        loss: Scalar loss value
        grad_logits: Gradient of loss with respect to logits
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Flatten for easier processing
    logits_flat = logits.reshape(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.reshape(-1)             # (batch_size * seq_len,)
    
    # Compute log softmax using log-sum-exp trick for numerical stability
    logits_max = np.max(logits_flat, axis=1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp
    
    # Get log probabilities for target tokens
    target_log_probs = log_probs[np.arange(len(targets_flat)), targets_flat]
    
    # Compute loss (negative log likelihood)
    loss = -np.mean(target_log_probs)
    
    # Compute gradient
    # Gradient of cross-entropy is: softmax(logits) - one_hot(targets)
    probs = np.exp(log_probs)
    grad_logits_flat = probs.copy()
    grad_logits_flat[np.arange(len(targets_flat)), targets_flat] -= 1
    grad_logits_flat /= len(targets_flat)  # Average over all positions
    
    # Reshape back to original shape
    grad_logits = grad_logits_flat.reshape(batch_size, seq_len, vocab_size)
    
    return loss, grad_logits


class TextDataset:
    """
    Character-level text dataset for language modeling.
    
    This class handles:
    - Character-to-index encoding
    - Creating training sequences
    - Batching data
    """
    
    def __init__(self, text: str, seq_len: int):
        """
        Initialize dataset.
        
        Args:
            text: Training text
            seq_len: Length of sequences for training
        """
        # Get unique characters (vocabulary)
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create character-to-index and index-to-character mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Encode entire text
        self.data = np.array([self.char_to_idx[ch] for ch in text])
        self.seq_len = seq_len
        
        print(f"Dataset initialized:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Total characters: {len(text):,}")
        print(f"  Sequence length: {seq_len}")
    
    def get_batch(self, batch_size: int, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of training data.
        
        Args:
            batch_size: Number of sequences in batch
            split: 'train' or 'val'
            
        Returns:
            inputs: Input sequences of shape (batch_size, seq_len)
            targets: Target sequences of shape (batch_size, seq_len)
                    (shifted by 1 position for next-token prediction)
        """
        # Split data into train/val (90/10 split)
        split_idx = int(0.9 * len(self.data))
        
        if split == 'train':
            data = self.data[:split_idx]
        else:
            data = self.data[split_idx:]
        
        # Random starting positions for each sequence in batch
        max_start = len(data) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        
        # Create input and target sequences
        inputs = np.array([data[i:i+self.seq_len] for i in starts])
        targets = np.array([data[i+1:i+self.seq_len+1] for i in starts])
        
        return inputs, targets
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text to indices."""
        return np.array([self.char_to_idx[ch] for ch in text])
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode indices to text."""
        return ''.join([self.idx_to_char[int(i)] for i in indices])


def train(
    model: GPT,
    dataset: TextDataset,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    eval_interval: int = 10,
    eval_samples: int = 5,
    save_path: str = 'checkpoints/gpt_model.pkl'
):
    """
    Train the GPT model.
    
    Args:
        model: GPT model to train
        dataset: Training dataset
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        eval_interval: How often to evaluate
        eval_samples: Number of batches to use for evaluation
        save_path: Where to save model checkpoints
    """
    print("\n" + "=" * 70)
    print("TRAINING GPT MODEL")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = AdamOptimizer(learning_rate=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training step
        inputs, targets = dataset.get_batch(batch_size, split='train')
        
        # Forward pass
        logits = model.forward(inputs)
        
        # Compute loss
        loss, grad_logits = cross_entropy_loss(logits, targets)
        
        # Backward pass
        model.backward(grad_logits)
        
        # Update parameters
        params = model.get_parameters()
        grads = model.get_gradients()
        optimizer.update(params, grads)
        
        train_losses.append(loss)
        
        # Evaluation
        if (epoch + 1) % eval_interval == 0:
            # Compute validation loss
            val_loss_sum = 0.0
            for _ in range(eval_samples):
                val_inputs, val_targets = dataset.get_batch(batch_size, split='val')
                val_logits = model.forward(val_inputs)
                val_loss, _ = cross_entropy_loss(val_logits, val_targets)
                val_loss_sum += val_loss
            
            avg_val_loss = val_loss_sum / eval_samples
            val_losses.append(avg_val_loss)
            
            # Compute time statistics
            elapsed = time.time() - start_time
            epochs_per_sec = (epoch + 1) / elapsed
            eta = (num_epochs - epoch - 1) / epochs_per_sec if epochs_per_sec > 0 else 0
            
            # Print progress
            print(f"Epoch {epoch+1:4d}/{num_epochs} | "
                  f"Train Loss: {loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"ETA: {eta:.0f}s")
            
            # Generate sample text
            if (epoch + 1) % (eval_interval * 5) == 0:
                sample_text = generate_sample(model, dataset, prompt="The ", length=100)
                print(f"Sample: {sample_text}")
                print()
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    
    return train_losses, val_losses


def generate_sample(model: GPT, dataset: TextDataset, prompt: str = "", length: int = 100) -> str:
    """
    Generate text from the model.
    
    Args:
        model: Trained GPT model
        dataset: Dataset (for encoding/decoding)
        prompt: Starting text
        length: Number of characters to generate
        
    Returns:
        Generated text
    """
    # Encode prompt
    if prompt:
        tokens = dataset.encode(prompt)
    else:
        tokens = np.array([0])  # Start with first token
    
    tokens = tokens.reshape(1, -1)  # Add batch dimension
    
    # Generate
    generated = model.generate(tokens, max_new_tokens=length, temperature=0.8)
    
    # Decode
    text = dataset.decode(generated[0])
    
    return text


def main():
    """Main training script."""
    
    # Load training data
    print("Loading training data...")
    data_path = "data/shakespeare.txt"
    
    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please create a text file for training.")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    seq_len = 64
    dataset = TextDataset(text, seq_len)
    
    # Create model
    print("\nInitializing model...")
    model = GPT(
        vocab_size=dataset.vocab_size,
        d_model=128,          # Small model for faster training
        num_layers=4,         # 4 transformer blocks
        num_heads=4,          # 4 attention heads
        d_ff=512,             # Feed-forward dimension
        max_seq_len=seq_len,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.size for _, p in model.get_parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    train(
        model=model,
        dataset=dataset,
        num_epochs=500,
        batch_size=32,
        learning_rate=3e-4,
        eval_interval=10,
        eval_samples=5,
        save_path='checkpoints/gpt_model.pkl'
    )
    
    # Generate final samples
    print("\n" + "=" * 70)
    print("SAMPLE GENERATIONS")
    print("=" * 70)
    
    prompts = ["The ", "To be ", "What ", "I am "]
    for prompt in prompts:
        generated = generate_sample(model, dataset, prompt=prompt, length=200)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
        print("-" * 70)


if __name__ == "__main__":
    main()
