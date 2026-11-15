"""CPU integration checks for timing hooks and workload replay, without torch."""
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_record import Recorder, make_workload, parse_args, replay
from record_metrics import summarize


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Sequence:
    def __init__(self, seq_id, tokens, maximum):
        self.seq_id = seq_id
        self.num_tokens = len(tokens)
        self.num_prompt_tokens = len(tokens)
        self.num_cached_tokens = self.num_scheduled_tokens = 0
        self.max_tokens = maximum
        self.block_table = []
        self.is_finished = False

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens


class Blocks:
    def __init__(self):
        self.blocks = list(range(100))
        self.free_block_ids = deque(self.blocks)
        self.cached_blocks_to_reuse = 0

    def allocate(self, seq, cached_blocks):
        seq.block_table = [self.free_block_ids.popleft()]
        seq.num_cached_tokens = cached_blocks * 2

    def deallocate(self, seq):
        self.free_block_ids.extend(seq.block_table)
        seq.block_table = []
        seq.num_cached_tokens = 0


class Scheduler:
    def __init__(self, clock):
        self.clock = clock
        self.waiting, self.running = deque(), deque()
        self.block_manager = Blocks()
        self.max_num_batched_tokens = 2
        self.block_size = 2

    def add(self, seq):
        self.waiting.append(seq)

    def schedule(self):
        self.clock.sleep(0.001)
        if self.waiting:
            seq = self.waiting[0]
            if not seq.block_table:
                self.block_manager.allocate(seq, self.block_manager.cached_blocks_to_reuse)
            seq.num_scheduled_tokens = min(2, seq.num_tokens - seq.num_cached_tokens)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.waiting.popleft()
                self.running.append(seq)
            return [seq], True
        seq = self.running[0]
        seq.num_scheduled_tokens = 1
        return [seq], False

    def preempt(self, seq):
        self.running.remove(seq)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs, token_ids, is_prefill):
        self.clock.sleep(0.001)
        for seq in seqs:
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.num_tokens += 1
            if seq.num_completion_tokens == seq.max_tokens:
                seq.is_finished = True
                self.running.remove(seq)
                self.block_manager.deallocate(seq)


class Runner:
    def __init__(self, clock):
        self.clock = clock
        self.fail = False

    def run(self, seqs, is_prefill):
        self.clock.sleep(0.01)
        if self.fail:
            raise RuntimeError('simulated GPU failure')
        return [7] * len(seqs)

    def call(self, method, *args):
        return getattr(self, method)(*args)


class Engine:
    def __init__(self, clock):
        self.scheduler = Scheduler(clock)
        self.model_runner = Runner(clock)
        self.next_id = 0

    def add_request(self, tokens, maximum):
        self.scheduler.add(Sequence(self.next_id, tokens, maximum))
        self.next_id += 1

    def is_finished(self):
        return not self.scheduler.waiting and not self.scheduler.running

    def step(self):
        seqs, prefill = self.scheduler.schedule()
        tokens = self.model_runner.call('run', seqs, prefill)
        self.scheduler.postprocess(seqs, tokens, prefill)


def workload(arrivals=(0.0,), maximum=2):
    return {'requests': [{'request_id': f'r{i}', 'cohort': 'short', 'arrival_s': arrival,
                          'prompt_token_ids': [1, 2, 3], 'max_tokens': maximum}
                         for i, arrival in enumerate(arrivals)]}


class RecordingTests(unittest.TestCase):
    def test_partial_prefill_has_no_phantom_tokens_and_hooks_restore(self):
        clock = Clock()
        engine = Engine(clock)
        data = workload()
        recorder = Recorder(data, clock)
        recorder.origin = clock()
        replay(engine, data, recorder, lambda count: count, 10, clock, clock.sleep)
        self.assertEqual([step['valid_output_tokens'] for step in recorder.steps], [0, 1, 1])
        self.assertEqual([event['event'] for event in recorder.events], ['enqueue', 'token', 'token', 'finish'])
        self.assertEqual(recorder.requests[0]['status'], 'completed')
        self.assertEqual(recorder.state(engine.scheduler)['kv_free'], 100)
        self.assertNotIn('run', vars(engine.model_runner))
        self.assertNotIn('schedule', vars(engine.scheduler))
        summary = summarize(recorder.requests, recorder.steps, clock())
        self.assertEqual(summary['counts']['output_tokens'], 2)
        self.assertEqual(summary['latency_ms']['itl_pooled']['count'], 1)

    def test_blocked_injection_is_included_in_arrival_ttft(self):
        clock = Clock()
        engine = Engine(clock)
        data = workload((0, 0.005))
        recorder = Recorder(data, clock)
        recorder.origin = clock()
        replay(engine, data, recorder, lambda count: count, 10, clock, clock.sleep)
        second = recorder.requests[1]
        self.assertGreater(second['enqueued_s'], second['arrival_s'])
        first = second['token_times_s'][0]
        self.assertGreater(first - second['arrival_s'], first - second['enqueued_s'])

    def test_preemption_counts_only_reexecuted_positions(self):
        clock = Clock()
        engine = Engine(clock)
        data = workload()
        recorder = Recorder(data, clock)
        recorder.origin = clock()
        with recorder.attach(engine):
            recorder.admitting = recorder.requests[0]
            engine.add_request([1, 2, 3], 2)
            recorder.step(engine)
            recorder.step(engine)
            seq = engine.scheduler.running[0]
            engine.scheduler.preempt(seq)
            while not engine.is_finished():
                recorder.step(engine)
        record = recorder.requests[0]
        self.assertEqual(record['preemptions'], 1)
        self.assertEqual(record['recomputed_tokens'], 3)
        self.assertEqual(len(record['token_times_s']), 2)

    def test_prefix_hit_after_preemption_avoids_recompute(self):
        clock = Clock()
        engine = Engine(clock)
        data = workload()
        recorder = Recorder(data, clock)
        recorder.origin = clock()
        with recorder.attach(engine):
            recorder.admitting = recorder.requests[0]
            engine.add_request([1, 2, 3], 2)
            recorder.step(engine)
            recorder.step(engine)
            engine.scheduler.preempt(engine.scheduler.running[0])
            engine.scheduler.block_manager.cached_blocks_to_reuse = 1
            while not engine.is_finished():
                recorder.step(engine)
        self.assertEqual(recorder.requests[0]['recomputed_tokens'], 1)
        self.assertEqual(recorder.requests[0]['cache_hit_tokens'], 2)

    def test_failed_step_records_elapsed_time_and_restores_hooks(self):
        clock = Clock()
        engine = Engine(clock)
        engine.model_runner.fail = True
        data = workload()
        recorder = Recorder(data, clock)
        recorder.origin = clock()
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            replay(engine, data, recorder, lambda count: count, 10, clock, clock.sleep)
        self.assertTrue(recorder.steps[0]['failed'])
        self.assertGreater(recorder.steps[0]['model_s'], 0)
        self.assertIsNone(recorder.steps[0]['postprocess_s'])
        self.assertNotIn('run', vars(engine.model_runner))
        self.assertNotIn('postprocess', vars(engine.scheduler))

    def test_initial_idle_and_cooperative_timeout(self):
        clock = Clock()
        engine = Engine(clock)
        data = workload((0.03,))
        recorder = Recorder(data, clock)
        recorder.origin = clock()
        replay(engine, data, recorder, lambda count: count, 10, clock, clock.sleep)
        self.assertAlmostEqual(recorder.requests[0]['enqueued_s'], 0.03)
        self.assertGreater(clock(), 0.03)
        delayed = Recorder(workload((100,)), clock)
        delayed.origin = clock()
        with self.assertRaises(TimeoutError):
            replay(engine, workload((100,)), delayed, lambda count: count, 0.05, clock, clock.sleep)
        self.assertIsNone(delayed.requests[0]['enqueued_s'])

    def test_trace_generation_is_reproducible(self):
        args = parse_args(['--requests', '4', '--short-input', '3', '--long-input', '4', '--output-tokens', '2'])
        self.assertEqual(make_workload(args), make_workload(args))
        self.assertEqual([len(row['prompt_token_ids']) for row in make_workload(args)['requests']], [3, 4, 3, 4])

    def test_invalid_graph_capacity_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(['--cuda-graph', '--max-num-seqs', '12'])


if __name__ == '__main__':
    unittest.main()
