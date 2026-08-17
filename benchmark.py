"""
Benchmarking & Profiling Suite for NumPy Transformer

Measures:
- Parameter counts & memory footprint
- Theoretical FLOPs per forward/backward pass (Kaplan et al. scaling laws)
- Throughput (tokens/sec) for training and inference
- Micro-benchmark latency breakdown per component (Embeddings, Attention, MLP, Norm)
"""

import time
import argparse
from typing import Dict, Any, Optional
import numpy as np

from gpt_numpy import GPT, MultiHeadAttention, FeedForward, LayerNorm
MLP = FeedForward
from layers import RMSNorm, SwiGLU
from train import cross_entropy_loss
from optimizers import AdamW


def count_parameters(model: GPT) -> int:
    """Calculate total trainable parameter count."""
    params = model.get_parameters()
    return sum(p.size for _, p in params)


def estimate_flops_per_token(
    vocab_size: int,
    n_layer: int,
    d_model: int,
    seq_len: int,
    num_heads: int,
    mlp_hidden_dim: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate theoretical FLOPs per forward token using standard transformer formula:
        Attention: 2 * (4 * d_model^2) + 2 * (d_model * seq_len) + 2 * (d_model * seq_len) = 8 * d_model^2 + 4 * d_model * seq_len
        MLP: 2 * (2 * d_model * hidden_dim) = 4 * d_model * hidden_dim (or 6 for SwiGLU)
        Logits: 2 * d_model * vocab_size
    """
    hidden_dim = mlp_hidden_dim or (4 * d_model)
    attn_flops = 8 * (d_model ** 2) + 4 * d_model * seq_len
    mlp_flops = 4 * d_model * hidden_dim
    layer_flops = attn_flops + mlp_flops
    total_model_flops = n_layer * layer_flops + (2 * d_model * vocab_size)
    
    # Backward pass is approximately 2x forward FLOPs
    training_flops_per_token = 3 * total_model_flops
    
    return {
        "forward_flops_per_token": total_model_flops,
        "backward_flops_per_token": 2 * total_model_flops,
        "training_flops_per_token": training_flops_per_token,
        "layer_flops": layer_flops
    }


def benchmark_throughput(
    vocab_size: int = 256,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    seq_len: int = 64,
    batch_size: int = 8,
    warmup_iters: int = 3,
    benchmark_iters: int = 10
) -> Dict[str, Any]:
    """Profile inference and training forward/backward throughput."""
    print("=" * 60)
    print(f"Profiling GPT [Layers: {num_layers}, Dim: {d_model}, Heads: {num_heads}, Seq: {seq_len}, Batch: {batch_size}]")
    print("=" * 60)
    
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=4 * d_model,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    num_params = count_parameters(model)
    print(f"Total Parameters: {num_params:,} ({num_params * 4 / (1024**2):.2f} MB float32)")
    
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # Warmup
    for _ in range(warmup_iters):
        logits = model.forward(x)
        
    # Benchmark Inference
    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        logits = model.forward(x)
    t1 = time.perf_counter()
    
    infer_time = (t1 - t0) / benchmark_iters
    total_tokens = batch_size * seq_len
    infer_throughput = total_tokens / infer_time
    
    # Benchmark Training step (Forward + Backward)
    for _ in range(warmup_iters):
        logits = model.forward(x)
        loss, grad_logits = cross_entropy_loss(logits, targets)
        model.backward(grad_logits)
        
    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        logits = model.forward(x)
        loss, grad_logits = cross_entropy_loss(logits, targets)
        model.backward(grad_logits)
    t1 = time.perf_counter()
    
    train_time = (t1 - t0) / benchmark_iters
    train_throughput = total_tokens / train_time
    
    flops_info = estimate_flops_per_token(vocab_size, num_layers, d_model, seq_len, num_heads)
    gflops_achieved = (flops_info["training_flops_per_token"] * train_throughput) / 1e9
    
    print(f"\n--- Benchmark Results ---")
    print(f"Inference Latency:   {infer_time * 1000:.2f} ms / step")
    print(f"Inference Throughput: {infer_throughput:.1f} tokens/sec")
    print(f"Training Latency:    {train_time * 1000:.2f} ms / step (Forward + Backward)")
    print(f"Training Throughput:  {train_throughput:.1f} tokens/sec")
    print(f"Achieved Compute:    {gflops_achieved:.3f} GFLOPs/s")
    print("=" * 60)
    
    return {
        "num_params": num_params,
        "infer_latency_ms": infer_time * 1000,
        "infer_throughput_tok_s": infer_throughput,
        "train_latency_ms": train_time * 1000,
        "train_throughput_tok_s": train_throughput,
        "gflops": gflops_achieved
    }


def benchmark_layer_breakdown(d_model: int = 128, num_heads: int = 4, seq_len: int = 64, batch_size: int = 8):
    """Profile latency of individual architectural subcomponents."""
    print("\n--- Microbenchmark: Layer Breakdown ---")
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    
    # 1. MultiHeadAttention
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = attn.forward(x)
    attn_time = (time.perf_counter() - t0) / 50 * 1000
    
    # 2. MLP
    mlp = MLP(d_model=d_model, d_ff=4 * d_model, dropout=0.0)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = mlp.forward(x)
    mlp_time = (time.perf_counter() - t0) / 50 * 1000
    
    # 3. SwiGLU
    swiglu = SwiGLU(d_model=d_model)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = swiglu.forward(x)
    swiglu_time = (time.perf_counter() - t0) / 50 * 1000
    
    # 4. LayerNorm
    ln = LayerNorm(normalized_shape=d_model)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = ln.forward(x)
    ln_time = (time.perf_counter() - t0) / 50 * 1000
    
    # 5. RMSNorm
    rmsnorm = RMSNorm(d_model=d_model)
    t0 = time.perf_counter()
    for _ in range(50):
        _ = rmsnorm.forward(x)
    rmsnorm_time = (time.perf_counter() - t0) / 50 * 1000
    
    print(f"Multi-Head Attention: {attn_time:.3f} ms")
    print(f"Standard GELU MLP:    {mlp_time:.3f} ms")
    print(f"Modern SwiGLU FFN:    {swiglu_time:.3f} ms")
    print(f"Standard LayerNorm:   {ln_time:.3f} ms")
    print(f"RMSNorm:              {rmsnorm_time:.3f} ms ({((ln_time - rmsnorm_time) / ln_time) * 100:+.1f}% vs LayerNorm)")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NumPy Transformer Benchmark")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    benchmark_throughput(
        d_model=args.d_model,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
    benchmark_layer_breakdown(
        d_model=args.d_model,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
