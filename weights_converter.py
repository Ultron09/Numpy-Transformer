"""
Weight Converter & Serialization Utilities for NumPy Transformer

Provides:
- Export/import model weights to/from standard dictionary formats
- Inspection and verification of weight shapes
- Weight export to binary/numpy formats
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pickle
import json
import os

from gpt_numpy import GPT


def extract_state_dict(model: GPT) -> Dict[str, np.ndarray]:
    """
    Extract all trainable model weights into a named dictionary.
    """
    params = model.get_parameters()
    state_dict = {}
    for name, array in params:
        state_dict[name] = np.copy(array)
    return state_dict


def load_state_dict(model: GPT, state_dict: Dict[str, np.ndarray], strict: bool = True) -> None:
    """
    Load weights from state_dict into model parameters.
    """
    params = dict(model.get_parameters())
    
    for name, param in params.items():
        if name in state_dict:
            src = state_dict[name]
            if src.shape != param.shape:
                raise ValueError(f"Shape mismatch for {name}: expected {param.shape}, got {src.shape}")
            np.copyto(param, src)
        elif strict:
            raise KeyError(f"Missing parameter in state_dict: {name}")


def export_npz(model: GPT, filepath: str) -> None:
    """Export model weights to compressed .npz archive."""
    state_dict = extract_state_dict(model)
    config = {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "num_layers": len(model.blocks),
        "num_heads": model.blocks[0].attention.num_heads if model.blocks else 1,
        "max_seq_len": model.max_seq_len,
    }
    np.savez_compressed(filepath, config=np.array([json.dumps(config)]), **state_dict)
    print(f"Model exported successfully to {filepath}")


def load_npz(filepath: str) -> Tuple[GPT, Dict[str, Any]]:
    """Load model from .npz archive."""
    data = np.load(filepath, allow_pickle=True)
    config_json = str(data["config"][0])
    config = json.loads(config_json)
    
    model = GPT(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"]
    )
    
    state_dict = {k: data[k] for k in data.files if k != "config"}
    load_state_dict(model, state_dict)
    return model, config
