"""
Example Usage of GPT Model

This script demonstrates:
1. Loading/creating a model
2. Training on sample text
3. Generating text from trained model
4. Visualizing attention patterns (optional)
"""

import numpy as np
from gpt_numpy import GPT
from train import TextDataset, generate_sample
import os


def main():
    """Main example script."""
    
    print("=" * 70)
    print("GPT MODEL - QUICK EXAMPLE")
    print("=" * 70)
    
    # Load or create sample text
    data_path = "data/shakespeare.txt"
    
    if not os.path.exists(data_path):
        # Create simple sample text if Shakespeare not available
        sample_text = """
        The quick brown fox jumps over the lazy dog.
        Machine learning is the science of getting computers to learn.
        Deep learning uses neural networks with multiple layers.
        Transformers are a type of neural network architecture.
        Attention mechanisms allow models to focus on relevant information.
        """ * 20  # Repeat to have enough data
        
        os.makedirs("data", exist_ok=True)
        with open(data_path, 'w') as f:
            f.write(sample_text)
        
        text = sample_text
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    print(f"\n1. LOADING DATA")
    print(f"   Text length: {len(text):,} characters")
    
    # Create dataset
    seq_len = 32
    dataset = TextDataset(text, seq_len)
    print(f"   Vocabulary size: {dataset.vocab_size}")
    
    # Create small model for quick demonstration
    print(f"\n2. CREATING MODEL")
    model = GPT(
        vocab_size=dataset.vocab_size,
        d_model=64,           # Very small for quick demo
        num_layers=2,         # Just 2 layers
        num_heads=4,          # 4 attention heads
        d_ff=256,             # Feed-forward dimension
        max_seq_len=seq_len,
        dropout=0.0           # No dropout for demo
    )
    
    total_params = sum(p.size for _, p in model.get_parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Generate before training
    print(f"\n3. GENERATION BEFORE TRAINING")
    prompt = "The "
    before_training = generate_sample(model, dataset, prompt=prompt, length=100)
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: {before_training[:100]}...")
    print(f"   (Random gibberish, as expected)")
    
    # Quick training
    print(f"\n4. TRAINING MODEL (50 epochs)")
    print(f"   This will take a moment...")
    
    from train import AdamOptimizer, cross_entropy_loss
    
    optimizer = AdamOptimizer(learning_rate=1e-3)
    batch_size = 16
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # Get batch
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
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    print(f"   Training complete!")
    
    # Generate after training
    print(f"\n5. GENERATION AFTER TRAINING")
    after_training = generate_sample(model, dataset, prompt=prompt, length=200)
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated:\n   {after_training}")
    
    # Try different prompts
    print(f"\n6. MORE GENERATIONS")
    test_prompts = ["To be ", "What ", "And "]
    for test_prompt in test_prompts:
        generated = generate_sample(model, dataset, prompt=test_prompt, length=150)
        print(f"\n   Prompt: '{test_prompt}'")
        print(f"   Generated: {generated}")
    
    # Save model
    print(f"\n7. SAVING MODEL")
    save_path = "checkpoints/example_model.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"   Model saved to {save_path}")
    
    # Demonstrate loading
    print(f"\n8. LOADING MODEL")
    new_model = GPT(
        vocab_size=dataset.vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_seq_len=seq_len,
        dropout=0.0
    )
    new_model.load(save_path)
    print(f"   Model loaded successfully!")
    
    # Verify loaded model works
    loaded_gen = generate_sample(new_model, dataset, prompt="The ", length=100)
    print(f"   Generated from loaded model: {loaded_gen}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Observations:")
    print("- The model learns patterns from the training text")
    print("- Generated text should mimic the style/structure of training data")
    print("- Longer training = better results")
    print("- Larger model = more capacity to learn complex patterns")
    print("\nNext Steps:")
    print("- Run 'python train.py' for full training")
    print("- Experiment with different hyperparameters")
    print("- Try your own training data")
    print("=" * 70)


if __name__ == "__main__":
    main()
