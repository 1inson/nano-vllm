"""Run the actual allocation method against simulated unequal GPU capacities.

Extracting the method avoids importing CUDA-only model dependencies on CPU CI.
The collective and tensor allocation are simulated, not NCCL integration tests.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def allocate_method(torch, dist):
    path = Path(__file__).resolve().parents[1] / 'nanovllm/engine/model_runner.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelRunner')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'allocate_kv_cache')
    module = ast.Module(body=[method], type_ignores=[])
    scope = {'torch': torch, 'dist': dist}
    exec(compile(module, str(path), 'exec'), scope)
    return scope['allocate_kv_cache']


def scenario(remote_capacity, *, local_total=1000000, world_size=2):
    events = []

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    def tensor(value, **kwargs):
        events.append(('capacity', value, kwargs))
        return Scalar(value)

    def all_reduce(value, op):
        events.append(('collective', op))
        value.value = min(value.value, remote_capacity)

    def empty(*shape):
        events.append(('allocate', shape))
        return shape

    fake_torch = SimpleNamespace(
        int64='int64', tensor=tensor, empty=empty,
        cuda=SimpleNamespace(mem_get_info=lambda: (local_total, local_total),
                             memory_stats=lambda: {'allocated_bytes.all.peak': 0, 'allocated_bytes.all.current': 0}))
    fake_dist = SimpleNamespace(all_reduce=all_reduce, ReduceOp=SimpleNamespace(MIN='MIN'))
    hf = SimpleNamespace(num_key_value_heads=4, hidden_size=16, num_attention_heads=4,
                         head_dim=4, num_hidden_layers=1, dtype=SimpleNamespace(itemsize=2))
    runner = SimpleNamespace(config=SimpleNamespace(hf_config=hf, gpu_memory_utilization=0.5),
                             block_size=256, world_size=world_size, rank=0,
                             model=SimpleNamespace(modules=lambda: []))
    return allocate_method(fake_torch, fake_dist), runner, events


class CapacityTests(unittest.TestCase):
    def test_different_ranks_use_minimum_capacity_for_pool_shape(self):
        method, runner, events = scenario(5)
        method(runner)
        self.assertEqual(runner.config.num_kvcache_blocks, 5)
        self.assertEqual(runner.kv_cache[2], 5)
        self.assertLess(next(i for i, event in enumerate(events) if event[0] == 'collective'),
                        next(i for i, event in enumerate(events) if event[0] == 'allocate'))
        self.assertEqual(events[0][2]['device'], 'cuda:0')

    def test_nonpositive_local_capacity_still_enters_collective_before_assert(self):
        method, runner, events = scenario(5, local_total=0)
        with self.assertRaises(AssertionError):
            method(runner)
        self.assertIn(('collective', 'MIN'), events)
        self.assertFalse(any(event[0] == 'allocate' for event in events))

    def test_single_gpu_does_not_add_collective(self):
        method, runner, events = scenario(0, world_size=1)
        method(runner)
        self.assertGreater(runner.config.num_kvcache_blocks, 0)
        self.assertFalse(any(event[0] == 'collective' for event in events))


if __name__ == '__main__':
    unittest.main()
