"""
Unit tests for weights_converter.py serialization and profiling
"""

import unittest
import numpy as np
import tempfile
import os

from weights_converter import (
    save_safetensors_binary,
    load_safetensors_binary,
    summarize_model_parameters
)


class TestWeightsConverter(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.state_dict = {
            "token_embeddings": np.random.randn(50, 32).astype(np.float32),
            "layer_1.weight": np.random.randn(32, 64).astype(np.float32),
            "layer_1.bias": np.zeros(64, dtype=np.float32),
        }
        self.metadata = {
            "model_name": "TestGPT",
            "version": "1.0.0",
            "author": "NumPy Transformer"
        }

    def test_safetensors_binary_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".npytens", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            save_safetensors_binary(self.state_dict, tmp_path, metadata=self.metadata)
            loaded_dict, loaded_meta = load_safetensors_binary(tmp_path)
            
            self.assertEqual(loaded_meta["model_name"], "TestGPT")
            self.assertEqual(set(loaded_dict.keys()), set(self.state_dict.keys()))
            
            for k in self.state_dict:
                np.testing.assert_allclose(loaded_dict[k], self.state_dict[k])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_model_summary(self):
        summary = summarize_model_parameters(self.state_dict)
        self.assertIn("total_parameters", summary)
        self.assertIn("memory_mb", summary)
        self.assertIn("global_sparsity", summary)
        
        expected_params = (50 * 32) + (32 * 64) + 64
        self.assertEqual(summary["total_parameters"], expected_params)
        self.assertTrue(summary["global_sparsity"] > 0.0)  # because bias is zeros


if __name__ == '__main__':
    unittest.main()
