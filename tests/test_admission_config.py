"""Validate admission settings without importing model or CUDA dependencies."""
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_record import parse_args


def load_config():
    """Load the real dataclass while temporarily supplying local HF metadata."""
    transformers = ModuleType('transformers')

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, model):
            return SimpleNamespace(max_position_embeddings=4096)

    transformers.AutoConfig = AutoConfig
    name = '_admission_config_under_test'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'nanovllm/config.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'transformers': transformers, name: module}):
        spec.loader.exec_module(module)
    return module.Config


class AdmissionConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='nano_admission_', dir=ROOT / 'tests')
        self.addCleanup(self.directory.cleanup)
        self.Config = load_config()

    def test_existing_callers_keep_unlimited_admission(self):
        config = self.Config(model=self.directory.name)
        self.assertIsNone(config.max_active_requests)

    def test_cap_is_independent_of_per_step_batch_limit(self):
        config = self.Config(model=self.directory.name, max_num_seqs=1,
                             max_active_requests=16)
        self.assertEqual(config.max_active_requests, 16)
        self.assertEqual(config.max_num_seqs, 1)

    def test_invalid_caps_are_rejected(self):
        for value in (0, -1, True, False, 1.5, '16'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'max_active_requests'):
                    self.Config(model=self.directory.name, max_active_requests=value)

    def test_temporary_loader_restores_dependency_modules(self):
        before = sys.modules.get('transformers')
        load_config()
        self.assertIs(sys.modules.get('transformers'), before)
        self.assertNotIn('_admission_config_under_test', sys.modules)


class AdmissionCliTests(unittest.TestCase):
    def test_default_keeps_existing_admission_policy(self):
        self.assertIsNone(parse_args([]).max_active_requests)

    def test_enable_sixteen_resident_requests(self):
        args = parse_args(['--max-active-requests', '16'])
        self.assertEqual(args.max_active_requests, 16)
        self.assertEqual(args.max_num_seqs, 16)

    def test_nonpositive_and_noninteger_caps_fail_at_parse_time(self):
        for value in ('0', '-1', '1.5', 'None'):
            with self.subTest(value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    parse_args(['--max-active-requests', value])
                self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
