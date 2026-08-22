"""
Model Quantization and Compression Toolkit for NumPy Transformers

Provides pure NumPy implementations of:
- Symmetric INT8 Weight Quantization (W8A32 & W8A8 dynamic quantization)
- Asymmetric UINT8 Quantization with zero-point offset
- QuantizedLinear layer storing weights in 8-bit integer precision (4x weight compression)
- Unstructured Magnitude Pruning with binary sparsity masks
- Model-wide Quantization and Memory Footprint Profiling
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def quantize_symmetric_int8(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Symmetrically quantize a 32-bit floating point tensor to signed 8-bit integers ([-127, 127]).
    
    Mathematical Formulation:
        scale = max(|tensor|) / 127.0
        q_tensor = clip(round(tensor / scale), -127, 127).astype(int8)
        
    Args:
        tensor: Floating point NumPy array
        
    Returns:
        Tuple of (quantized_int8_array, float_scale_factor)
    """
    max_val = float(np.max(np.abs(tensor)))
    if max_val == 0.0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
        
    scale = max_val / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, scale


def dequantize_symmetric_int8(q_tensor: np.ndarray, scale: float) -> np.ndarray:
    """
    Dequantize an INT8 array back to FP32 using its scaling factor.
    
    Mathematical Formulation:
        tensor ≈ q_tensor * scale
    """
    return q_tensor.astype(np.float32) * scale


def quantize_asymmetric_uint8(tensor: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Asymmetrically quantize a floating point tensor to unsigned 8-bit integers ([0, 255]).
    
    Mathematical Formulation:
        scale = (max(tensor) - min(tensor)) / 255.0
        zero_point = round(-min(tensor) / scale)
        q_tensor = clip(round(tensor / scale) + zero_point, 0, 255).astype(uint8)
        
    Args:
        tensor: Floating point NumPy array
        
    Returns:
        Tuple of (quantized_uint8_array, float_scale_factor, int_zero_point)
    """
    min_val = float(np.min(tensor))
    max_val = float(np.max(tensor))
    
    if max_val == min_val:
        return np.zeros_like(tensor, dtype=np.uint8), 1.0, 0
        
    scale = (max_val - min_val) / 255.0
    zero_point = int(np.clip(np.round(-min_val / scale), 0, 255))
    quantized = np.clip(np.round(tensor / scale) + zero_point, 0, 255).astype(np.uint8)
    return quantized, scale, zero_point


def dequantize_asymmetric_uint8(q_tensor: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """
    Dequantize a UINT8 array back to FP32 using scale and zero_point.
    
    Mathematical Formulation:
        tensor ≈ (q_tensor - zero_point) * scale
    """
    return (q_tensor.astype(np.float32) - zero_point) * scale


class QuantizedLinear:
    """
    Linear layer storing weights in INT8 precision for 75% memory reduction.
    
    Supports:
    - W8A32: INT8 weights dequantized dynamically during forward computation
    - W8A8: Fully integer dot-product with quantized activations
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Stored in INT8 format: (out_features, in_features)
        self.q_weight = np.zeros((out_features, in_features), dtype=np.int8)
        self.weight_scale = 1.0
        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None
        
    @classmethod
    def from_float_linear(cls, linear_layer) -> "QuantizedLinear":
        """Construct a QuantizedLinear instance from a trained FP32 Linear layer."""
        out_features, in_features = linear_layer.weight.shape
        has_bias = linear_layer.bias is not None
        
        ql = cls(in_features, out_features, bias=has_bias)
        ql.q_weight, ql.weight_scale = quantize_symmetric_int8(linear_layer.weight)
        
        if has_bias:
            ql.bias = linear_layer.bias.copy().astype(np.float32)
            
        return ql
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using W8A32 dynamic dequantization.
        
        Args:
            x: Input array of shape (..., in_features)
            
        Returns:
            Output array of shape (..., out_features)
        """
        w_fp32 = dequantize_symmetric_int8(self.q_weight, self.weight_scale)
        output = x @ w_fp32.T
        
        if self.bias is not None:
            output += self.bias
            
        return output
        
    def forward_int8_dynamic(self, x: np.ndarray) -> np.ndarray:
        """
        W8A8 forward pass: Quantizes activations dynamically at runtime and computes
        integer matrix multiplication.
        """
        # Quantize input activation
        x_q, x_scale = quantize_symmetric_int8(x)
        
        # Integer matrix multiplication (promoted to int32 to prevent overflow)
        # x_q shape: (..., in_features), q_weight shape: (out_features, in_features)
        int_acc = np.matmul(x_q.astype(np.int32), self.q_weight.astype(np.int32).T)
        
        # Scale back to FP32: output = int_acc * (x_scale * w_scale)
        output = int_acc.astype(np.float32) * (x_scale * self.weight_scale)
        
        if self.bias is not None:
            output += self.bias
            
        return output


def magnitude_prune(weights: np.ndarray, sparsity: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply magnitude pruning to a weight tensor to set smallest absolute weights to zero.
    
    Args:
        weights: Weight array of any shape
        sparsity: Target fraction of weights to prune (0.0 to 1.0)
        
    Returns:
        Tuple of (pruned_weights, binary_boolean_mask)
    """
    if sparsity <= 0.0:
        return weights.copy(), np.ones_like(weights, dtype=bool)
    if sparsity >= 1.0:
        return np.zeros_like(weights), np.zeros_like(weights, dtype=bool)
        
    kth_percentile = sparsity * 100.0
    threshold = float(np.percentile(np.abs(weights), kth_percentile))
    
    mask = np.abs(weights) >= threshold
    pruned_weights = weights * mask
    return pruned_weights, mask


def profile_model_memory(model: Any) -> Dict[str, Any]:
    """
    Profile the memory footprint of all parameters in a transformer model.
    
    Returns:
        Dictionary with total parameter count, memory in bytes/MB, and layer breakdowns.
    """
    total_bytes = 0
    total_params = 0
    breakdown = {}
    
    def inspect_obj(name: str, obj: Any):
        nonlocal total_bytes, total_params
        if isinstance(obj, np.ndarray):
            total_bytes += obj.nbytes
            total_params += obj.size
            breakdown[name] = {"shape": obj.shape, "dtype": str(obj.dtype), "size_bytes": obj.nbytes}
        elif hasattr(obj, "__dict__"):
            for attr_name, attr_val in obj.__dict__.items():
                if not attr_name.startswith("_") and not callable(attr_val):
                    inspect_obj(f"{name}.{attr_name}", attr_val)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                inspect_obj(f"{name}[{i}]", item)
                
    inspect_obj("model", model)
    
    return {
        "total_parameters": total_params,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "layers": breakdown
    }
