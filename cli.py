"""
Interactive CLI & Playground for NumPy Transformer

Provides:
- Interactive text generation prompt REPL with streaming output
- Configurable sampling parameters (temperature, top-k, top-p, repetition penalty)
- Model inspection & architecture summary
"""

import sys
import os
import argparse
import time
import pickle
import numpy as np

from gpt_numpy import GPT
from sampler import GenerationSampler, generate_stream
from tokenizer import CharTokenizer


def load_model_and_tokenizer(checkpoint_path: str, data_path: str = "data/shakespeare.txt"):
    """Load model checkpoint and fit tokenizer."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Run example.py or train.py first.")
        
    with open(checkpoint_path, "rb") as f:
        model = pickle.load(f)
        
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokenizer = CharTokenizer().fit(text)
    return model, tokenizer


def print_banner(model: GPT):
    params = sum(p.size for _, p in model.get_parameters())
    print("\n" + "=" * 65)
    print(" 🚀 NumPy-Transformer Interactive Console")
    print("=" * 65)
    print(f" • Architecture:   {len(model.blocks)} Layers | {model.d_model} Dim | {model.blocks[0].attention.num_heads} Heads")
    print(f" • Context Window: {model.max_seq_len} tokens")
    print(f" • Total Params:   {params:,} ({params * 4 / (1024**2):.2f} MB float32)")
    print(" • Commands:       /temp <val>, /topk <val>, /topp <val>, /rep <val>, /exit")
    print("=" * 65 + "\n")


def repl(model: GPT, tokenizer: CharTokenizer):
    sampler = GenerationSampler(
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.15
    )
    
    print_banner(model)
    
    while True:
        try:
            prompt = input("\033[92mPrompt >\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not prompt:
            continue
            
        if prompt == "/exit":
            print("Goodbye!")
            break
            
        if prompt.startswith("/temp "):
            sampler.temperature = float(prompt.split()[1])
            print(f"Temperature set to {sampler.temperature}")
            continue
            
        if prompt.startswith("/topk "):
            sampler.top_k = int(prompt.split()[1])
            print(f"Top-K set to {sampler.top_k}")
            continue
            
        if prompt.startswith("/topp "):
            sampler.top_p = float(prompt.split()[1])
            print(f"Top-P set to {sampler.top_p}")
            continue
            
        if prompt.startswith("/rep "):
            sampler.repetition_penalty = float(prompt.split()[1])
            print(f"Repetition penalty set to {sampler.repetition_penalty}")
            continue
            
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]
            
        print(f"\033[94mGPT:\033[0m {prompt}", end="", flush=True)
        
        t0 = time.perf_counter()
        token_count = 0
        
        for token_id in generate_stream(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=150,
            sampler=sampler
        ):
            ch = tokenizer.decode([token_id])
            print(ch, end="", flush=True)
            token_count += 1
            
        elapsed = time.perf_counter() - t0
        speed = token_count / max(1e-4, elapsed)
        print(f"\n\033[90m[{token_count} tokens in {elapsed:.2f}s | {speed:.1f} tok/s]\033[0m\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive NumPy-Transformer Console")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/example_model.pkl")
    parser.add_argument("--data", type=str, default="data/shakespeare.txt")
    args = parser.parse_args()
    
    try:
        model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.data)
        repl(model, tokenizer)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train an example model first using: python example.py")
