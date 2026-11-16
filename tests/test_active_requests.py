"""CPU checks of real scheduler/KV lifecycles without importing CUDA packages.

Only xxhash and NumPy's byte conversion are replaced by deterministic stdlib
equivalents. These tests exercise allocation/sharing, not hash compatibility or
the numerical correctness of attention kernels.
"""
import hashlib
import importlib.util
from itertools import count
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def scheduler_types():
    """Load actual modules while keeping dependency stubs out of other tests."""
    package = ModuleType('nanovllm')
    package.__path__ = [str(ROOT / 'nanovllm')]
    engine = ModuleType('nanovllm.engine')
    engine.__path__ = [str(ROOT / 'nanovllm/engine')]
    config = ModuleType('nanovllm.config')
    config.Config = SimpleNamespace

    class Hash:
        def __init__(self):
            self.value = hashlib.blake2b(digest_size=8)

        def update(self, value):
            self.value.update(value)

        def intdigest(self):
            return int.from_bytes(self.value.digest(), 'little')

    def array(tokens):
        return SimpleNamespace(tobytes=lambda: struct.pack(f'<{len(tokens)}q', *tokens))

    dependencies = {
        'nanovllm': package,
        'nanovllm.engine': engine,
        'nanovllm.config': config,
        'xxhash': SimpleNamespace(xxh64=Hash),
        'numpy': SimpleNamespace(array=array),
    }
    modules = {}
    with patch.dict(sys.modules, dependencies):
        for suffix in ('sampling_params', 'engine.sequence',
                       'engine.block_manager', 'engine.scheduler'):
            name = f'nanovllm.{suffix}'
            path = ROOT / 'nanovllm' / (suffix.replace('.', '/') + '.py')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[suffix] = module
    return (modules['sampling_params'].SamplingParams,
            modules['engine.sequence'].Sequence,
            modules['engine.sequence'].SequenceStatus,
            modules['engine.scheduler'].Scheduler)


SamplingParams, Sequence, SequenceStatus, Scheduler = scheduler_types()


class ActiveRequestTests(unittest.TestCase):
    def setUp(self):
        # Small blocks allow real allocation/growth/preemption tests with tiny inputs.
        Sequence.block_size = 4
        Sequence.counter = count()

    def scheduler(self, limit=16, *, batch=16, budget=64, blocks=512):
        return Scheduler(SimpleNamespace(
            max_num_seqs=batch,
            max_active_requests=limit,
            max_num_batched_tokens=budget,
            eos=99,
            kvcache_block_size=Sequence.block_size,
            num_kvcache_blocks=blocks,
        ))

    def request(self, tokens, maximum=3, *, ignore_eos=True):
        return Sequence(tokens, SamplingParams(
            max_tokens=maximum, ignore_eos=ignore_eos))

    def assert_residents(self, scheduler, requests):
        resident_count = sum(bool(seq.block_table) for seq in requests)
        self.assertEqual(scheduler.num_active_requests, resident_count)
        if scheduler.max_active_requests is not None:
            self.assertLessEqual(resident_count, scheduler.max_active_requests)

    def step(self, scheduler, requests, token=7):
        scheduled, prefill = scheduler.schedule()
        self.assertTrue(scheduled)
        self.assert_residents(scheduler, requests)
        scheduler.postprocess(scheduled, [token] * len(scheduled), prefill)
        self.assert_residents(scheduler, requests)
        return scheduled, prefill

    def drain(self, scheduler, requests, maximum_steps=4096):
        for _ in range(maximum_steps):
            if scheduler.is_finished():
                break
            self.step(scheduler, requests)
        else:
            self.fail('scheduler did not finish within the step bound')
        self.assertTrue(all(seq.is_finished for seq in requests))
        self.assertEqual(scheduler.num_active_requests, 0)
        self.assertTrue(all(not seq.block_table for seq in requests))
        self.assertFalse(scheduler.block_manager.used_block_ids)
        self.assertTrue(all(block.ref_count == 0
                            for block in scheduler.block_manager.blocks))

    def test_burst_of_128_stays_within_global_limit_and_all_requests_finish(self):
        scheduler = self.scheduler()
        requests = [self.request([index + 1] * (3 if index % 2 == 0 else 4))
                    for index in range(128)]
        for seq in requests:
            scheduler.add(seq)

        scheduled, prefill = self.step(scheduler, requests)
        self.assertTrue(prefill)
        self.assertEqual(len(scheduled), 16)
        self.assertEqual(scheduler.num_active_requests, 16)
        self.assertEqual(len(scheduler.waiting), 112)
        self.assertTrue(all(not seq.block_table for seq in requests[16:]))

        scheduled, prefill = self.step(scheduler, requests)
        self.assertFalse(prefill)
        self.assertEqual(scheduled, requests[:16])
        self.drain(scheduler, requests)
        self.assertTrue(all(seq.num_completion_tokens == 3 for seq in requests))

    def test_partial_prefill_occupies_a_slot_and_can_continue_at_full_limit(self):
        scheduler = self.scheduler(1, budget=2, blocks=8)
        first = self.request(list(range(1, 8)), maximum=2)
        second = self.request([20], maximum=2)
        requests = [first, second]
        for seq in requests:
            scheduler.add(seq)

        scheduled, prefill = self.step(scheduler, requests)
        self.assertTrue(prefill)
        self.assertEqual(scheduled, [first])
        self.assertEqual(first.status, SequenceStatus.WAITING)
        self.assertEqual(first.num_cached_tokens, 2)
        self.assertEqual(first.num_completion_tokens, 0)
        self.assertEqual(scheduler.num_active_requests, 1)
        self.assertFalse(scheduler.running)
        self.assertFalse(second.block_table)

        # Each remaining chunk must progress despite the already occupied slot.
        for cached in (4, 6, 7):
            scheduled, prefill = self.step(scheduler, requests)
            self.assertEqual(scheduled, [first])
            self.assertTrue(prefill)
            self.assertEqual(first.num_cached_tokens, cached)
            self.assertFalse(second.block_table)
        self.assertEqual(first.num_completion_tokens, 1)
        self.assertEqual(first.status, SequenceStatus.RUNNING)
        self.drain(scheduler, requests, maximum_steps=16)

    def test_finishing_in_prefill_releases_slot_before_next_admission(self):
        scheduler = self.scheduler(1)
        first = self.request([1, 2, 3], maximum=1)
        second = self.request([4, 5, 6], maximum=1)
        requests = [first, second]
        for seq in requests:
            scheduler.add(seq)

        self.step(scheduler, requests)
        self.assertTrue(first.is_finished)
        self.assertEqual(scheduler.num_active_requests, 0)
        self.assertFalse(second.block_table)
        self.drain(scheduler, requests)

    def test_eos_completion_releases_slot(self):
        scheduler = self.scheduler(1)
        first = self.request([1, 2], maximum=8, ignore_eos=False)
        second = self.request([3, 4], maximum=2)
        requests = [first, second]
        for seq in requests:
            scheduler.add(seq)

        self.step(scheduler, requests, token=99)
        self.assertTrue(first.is_finished)
        self.assertEqual(first.num_completion_tokens, 1)
        self.assertEqual(scheduler.num_active_requests, 0)
        self.assertFalse(second.block_table)
        self.drain(scheduler, requests)

    def test_kv_growth_preemption_releases_slot_and_request_can_resume(self):
        scheduler = self.scheduler(2, blocks=2)
        requests = [self.request([1, 2, 3]), self.request([4, 5, 6])]
        for seq in requests:
            scheduler.add(seq)

        preempted = []
        original_preempt = scheduler.preempt

        def record_preempt(seq):
            original_preempt(seq)
            preempted.append(seq)
            self.assertFalse(seq.block_table)
            self.assert_residents(scheduler, requests)

        scheduler.preempt = record_preempt
        self.drain(scheduler, requests, maximum_steps=32)
        self.assertEqual(preempted, [requests[1]])
        self.assertTrue(all(seq.num_completion_tokens == 3 for seq in requests))

    def test_requests_sharing_prefix_blocks_still_occupy_separate_slots(self):
        scheduler = self.scheduler(2, blocks=8)
        prompt = [1, 2, 3, 4, 5]
        first = self.request(prompt, maximum=2)
        second = self.request(prompt, maximum=2)
        third = self.request(prompt, maximum=2)
        requests = [first, second, third]
        scheduler.add(first)
        self.step(scheduler, requests)
        scheduler.add(second)
        scheduler.add(third)

        scheduled, prefill = self.step(scheduler, requests)
        self.assertTrue(prefill)
        self.assertEqual(scheduled, [second])
        self.assertEqual(first.block_table[0], second.block_table[0])
        shared = scheduler.block_manager.blocks[first.block_table[0]]
        self.assertEqual(shared.ref_count, 2)
        self.assertEqual(scheduler.num_active_requests, 2)
        self.assertFalse(third.block_table)
        self.drain(scheduler, requests)

    def test_global_limit_is_independent_of_per_step_batch_limit(self):
        scheduler = self.scheduler(3, batch=1)
        requests = [self.request([index + 1], maximum=2) for index in range(5)]
        for seq in requests:
            scheduler.add(seq)

        for index in range(3):
            scheduled, prefill = self.step(scheduler, requests)
            self.assertTrue(prefill)
            self.assertEqual(scheduled, [requests[index]])
            self.assertEqual(scheduler.num_active_requests, index + 1)
        scheduled, prefill = self.step(scheduler, requests)
        self.assertFalse(prefill)
        self.assertEqual(len(scheduled), 1)
        self.assertTrue(all(not seq.block_table for seq in requests[3:]))
        self.drain(scheduler, requests)

    def test_none_retains_unbounded_admission_across_prefill_batches(self):
        scheduler = self.scheduler(None)
        requests = [self.request([index + 1], maximum=2) for index in range(128)]
        for seq in requests:
            scheduler.add(seq)

        for _ in range(8):
            scheduled, prefill = self.step(scheduler, requests)
            self.assertTrue(prefill)
            self.assertEqual(len(scheduled), 16)
        self.assertFalse(scheduler.waiting)
        self.assertEqual(len(scheduler.running), 128)
        self.assertEqual(scheduler.num_active_requests, 128)
        self.drain(scheduler, requests)


if __name__ == '__main__':
    unittest.main()
