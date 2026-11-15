"""CPU-only checks for rental preflight; no torch import is required."""

import importlib.util
from pathlib import Path
import socket
import unittest


SPEC = importlib.util.spec_from_file_location("check_multi_gpu", Path(__file__).resolve().parents[1] / "check_multi_gpu.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class ModelChecksTests(unittest.TestCase):
    def config(self):
        return dict(model_type="qwen3", hidden_size=1025, num_attention_heads=16,
                    num_key_value_heads=8, head_dim=128, intermediate_size=3072,
                    vocab_size=151936)

    def test_explicit_head_dim_does_not_require_hidden_divisibility(self):
        checks = CHECK.model_checks(self.config(), 2)
        self.assertFalse([item for item in checks if item[0] == "FAIL"])

    def test_kv_heads_must_divide_tp(self):
        config = self.config()
        config["num_key_value_heads"] = 2
        checks = CHECK.model_checks(config, 4)
        self.assertTrue(any(level == "FAIL" and name == "num_key_value_heads" for level, name, _ in checks))

    def test_mlp_and_vocabulary_shards(self):
        config = self.config()
        config.update(intermediate_size=3073, vocab_size=151937)
        failed = {name for level, name, _ in CHECK.model_checks(config, 2) if level == "FAIL"}
        self.assertEqual(failed, {"intermediate_size", "vocab_size"})

    def test_unknown_architecture_is_rejected(self):
        config = self.config()
        config["model_type"] = "qwen3_moe"
        self.assertIn(("FAIL", "model_type", "this runner implements dense Qwen3 only"), CHECK.model_checks(config, 2))

    def test_invalid_dimension_is_rejected(self):
        config = self.config()
        config["num_attention_heads"] = 0
        self.assertTrue(any(level == "FAIL" and name == "num_attention_heads" for level, name, _ in CHECK.model_checks(config, 2)))

    def test_fallback_head_dim_requires_integral_width(self):
        config = self.config()
        del config["head_dim"]
        self.assertTrue(any(level == "FAIL" and name == "head_dim" for level, name, _ in CHECK.model_checks(config, 2)))

    def test_occupied_port_is_detected_without_stealing_it(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            port = server.getsockname()[1]
            available, _ = CHECK.can_bind("127.0.0.1", port)
            self.assertFalse(available)


if __name__ == "__main__":
    unittest.main()
