"""
Weight Converter & Serialization Utilities for NumPy Transformer

Provides:
- Export/import model weights to/from standard dictionary formats
- Inspection and verification of weight shapes
- Weight export to compressed .npz archives
- Binary zero-dependency Safetensors-style format with metadata and checksums
- Model parameter memory footprint and sparsity profiling
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import json
import struct
import hashlib
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


def save_safetensors_binary(state_dict: Dict[str, np.ndarray], filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save tensors in pure-NumPy binary format with zero-copy memory layout.
    
    File Structure:
    - [8 bytes]: Magic Header (b'NPYTENS\x00')
    - [8 bytes]: Unsigned 64-bit integer specifying header JSON length (N)
    - [N bytes]: UTF-8 JSON header metadata containing tensor offsets, shapes, and dtypes
    - [...]: Contiguous raw binary tensor buffers
    """
    metadata = metadata or {}
    header = {"__metadata__": metadata}
    
    current_offset = 0
    tensor_buffers = []
    
    for name, tensor in state_dict.items():
        arr = np.ascontiguousarray(tensor)
        raw_bytes = arr.tobytes()
        byte_len = len(raw_bytes)
        
        header[name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data_offsets": [current_offset, current_offset + byte_len],
            "sha256": hashlib.sha256(raw_bytes).hexdigest()[:16]
        }
        
        tensor_buffers.append(raw_bytes)
        current_offset += byte_len
        
    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)
    
    with open(filepath, 'wb') as f:
        # Magic bytes + header length (uint64)
        f.write(b'NPYTENS\x00')
        f.write(struct.pack('<Q', header_len))
        f.write(header_json)
        for buf in tensor_buffers:
            f.write(buf)


def load_safetensors_binary(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load tensors from binary safetensors format.
    """
    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic != b'NPYTENS\x00':
            raise ValueError(f"Invalid file format: magic bytes {magic} do not match b'NPYTENS\\x00'")
            
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        
        metadata = header.get("__metadata__", {})
        data_start = 16 + header_len
        
        state_dict = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
                
            start_off, end_off = meta["data_offsets"]
            f.seek(data_start + start_off)
            raw_bytes = f.read(end_off - start_off)
            
            arr = np.frombuffer(raw_bytes, dtype=np.dtype(meta["dtype"]))
            arr = arr.reshape(meta["shape"]).copy()
            state_dict[name] = arr
            
    return state_dict, metadata


def summarize_model_parameters(state_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Compute comprehensive parameter statistics, memory footprint, and sparsity.
    """
    total_params = 0
    total_bytes = 0
    zero_params = 0
    
    per_layer_stats = {}
    for name, tensor in state_dict.items():
        count = int(np.prod(tensor.shape))
        n_bytes = tensor.nbytes
        zeros = int(np.sum(tensor == 0))
        
        total_params += count
        total_bytes += n_bytes
        zero_params += zeros
        
        per_layer_stats[name] = {
            "shape": list(tensor.shape),
            "params": count,
            "bytes": n_bytes,
            "sparsity": float(zeros / count) if count > 0 else 0.0
        }
        
    return {
        "total_parameters": total_params,
        "memory_mb": float(total_bytes / (1024 * 1024)),
        "global_sparsity": float(zero_params / total_params) if total_params > 0 else 0.0,
        "layers": per_layer_stats
    }
