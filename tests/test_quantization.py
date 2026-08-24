"""
Unit Tests for Quantization and Pruning Toolkit
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantization import (
    quantize_symmetric_int8,
    dequantize_symmetric_int8,
    quantize_asymmetric_uint8,
    dequantize_asymmetric_uint8,
    QuantizedLinear,
    magnitude_prune,
    profile_model_memory
)
from gpt_numpy import Linear


class TestQuantization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_symmetric_int8_quantization(self):
        w = np.random.randn(32, 64).astype(np.float32)
        q_w, scale = quantize_symmetric_int8(w)
        
        self.assertEqual(q_w.dtype, np.int8)
        self.assertTrue(np.all(q_w >= -127) and np.all(q_w <= 127))
        
        deq_w = dequantize_symmetric_int8(q_w, scale)
        mae = np.mean(np.abs(w - deq_w))
        self.assertLess(mae, 0.05, f"INT8 quantization error too high: {mae}")

    def test_asymmetric_uint8_quantization(self):
        w = np.random.uniform(-2.0, 5.0, size=(16, 32)).astype(np.float32)
        q_w, scale, zero_point = quantize_asymmetric_uint8(w)
        
        self.assertEqual(q_w.dtype, np.uint8)
        self.assertTrue(np.all(q_w >= 0) and np.all(q_w <= 255))
        
        deq_w = dequantize_asymmetric_uint8(q_w, scale, zero_point)
        mae = np.mean(np.abs(w - deq_w))
        self.assertLess(mae, 0.05, f"UINT8 quantization error too high: {mae}")

    def test_quantized_linear_forward(self):
        float_linear = Linear(in_features=32, out_features=16)
        x = np.random.randn(4, 8, 32).astype(np.float32)
        
        float_out = float_linear.forward(x)
        
        quant_linear = QuantizedLinear.from_float_linear(float_linear)
        quant_out = quant_linear.forward(x)
        
        mae = np.mean(np.abs(float_out - quant_out))
        self.assertLess(mae, 0.05, f"QuantizedLinear output drifted excessively: {mae}")

    def test_magnitude_prune(self):
        w = np.random.randn(40, 50).astype(np.float32)
        sparsity_target = 0.60
        
        pruned_w, mask = magnitude_prune(w, sparsity=sparsity_target)
        
        actual_sparsity = 1.0 - (np.count_nonzero(pruned_w) / pruned_w.size)
        self.assertAlmostEqual(actual_sparsity, sparsity_target, places=2)
        self.assertTrue(np.all(pruned_w[~mask] == 0.0))


if __name__ == "__main__":
    unittest.main()
