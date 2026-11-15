"""Record reproducible engine-visible request/token/step metrics (not HTTP latency).

Linux GPU example:
  python bench_record.py --model ~/huggingface/Qwen3-0.6B --cuda-graph
  python bench_record.py --trace-in benchmark_records/RUN/workload.json --cuda-graph

CPU-only workload preparation:
  python bench_record.py --dry-run --output /path/to/new-directory

Synthetic token IDs measure performance, not answer quality. Events are buffered
in memory and written after timing; nvidia-smi sampling can be disabled with
--gpu-sample-interval 0. Run ONE Python process; the engine spawns TP workers.
"""

import argparse
import atexit
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import threading
import time
import traceback
import uuid

from record_metrics import summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='~/huggingface/Qwen3-0.6B')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--workload', choices=['short', 'long', 'mixed'], default='mixed')
    parser.add_argument('--short-input', type=int, default=128)
    parser.add_argument('--long-input', type=int, default=2048)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--arrival-rate', type=float, default=2.0,
                        help='requests/s; 0 means a simultaneous burst')
    parser.add_argument('--arrival-process', choices=['constant', 'poisson'], default='poisson')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trace-in', type=Path, help='replay a previously saved workload.json')
    parser.add_argument('--output', type=Path, help='new output directory; existing files are never overwritten')
    parser.add_argument('--tp', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--cuda-graph', action='store_true')
    parser.add_argument('--max-num-seqs', type=int, default=16)
    parser.add_argument('--token-budget', type=int, default=1024)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.60)
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=300,
                        help='cooperative deadline checked between engine steps; cannot interrupt a hung collective')
    parser.add_argument('--gpu-sample-interval', type=float, default=1.0,
                        help='nvidia-smi interval in seconds, 0 disables sampling')
    parser.add_argument('--slo-ttft-ms', type=float)
    parser.add_argument('--slo-itl-ms', type=float,
                        help='per-request maximum ITL limit for goodput')
    parser.add_argument('--dry-run', action='store_true', help='save workload without importing GPU dependencies')
    args = parser.parse_args(argv)
    for name in ['requests', 'short_input', 'long_input', 'output_tokens',
                 'max_num_seqs', 'token_budget', 'max_model_len']:
        if getattr(args, name) < 1:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    for name in ['arrival_rate', 'gpu_sample_interval', 'max_seconds']:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == 'max_seconds' and value == 0):
            parser.error('--' + name.replace('_', '-') + ' must be finite and in range')
    if not 0 < args.gpu_memory_utilization < 1 or args.warmup_runs < 0:
        parser.error('memory utilization must be in (0, 1); warmup runs must be nonnegative')
    if args.max_num_seqs > 512:
        parser.error('--max-num-seqs must be <= 512 for this benchmark')
    if args.cuda_graph and args.max_num_seqs not in [1, 2, 4, 8] and args.max_num_seqs % 16:
        parser.error('this upstream Graph implementation requires max-num-seqs of 1,2,4,8 or a multiple of 16')
    for name in ['slo_ttft_ms', 'slo_itl_ms']:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error('--' + name.replace('_', '-') + ' must be finite and positive')
    return args


def make_workload(args):
    if args.trace_in:
        workload = json.loads(args.trace_in.read_text(encoding='utf-8'))
    else:
        arrivals = random.Random(args.seed)
        prompts = random.Random(args.seed + 1000003)
        arrival = 0.0
        rows = []
        for index in range(args.requests):
            if index and args.arrival_rate:
                arrival += (arrivals.expovariate(args.arrival_rate)
                            if args.arrival_process == 'poisson' else 1 / args.arrival_rate)
            cohort = args.workload if args.workload != 'mixed' else ('short' if index % 2 == 0 else 'long')
            length = args.short_input if cohort == 'short' else args.long_input
            rows.append({'request_id': f'r{index:06d}', 'cohort': cohort,
                         'arrival_s': arrival, 'max_tokens': args.output_tokens,
                         'prompt_token_ids': [prompts.randint(100, 10000) for _ in range(length)]})
        workload = {'schema_version': 1, 'synthetic': True, 'seed': args.seed,
                    'arrival_process': args.arrival_process, 'arrival_rate': args.arrival_rate,
                    'requests': rows}
    rows = workload.get('requests')
    if not isinstance(rows, list) or not rows:
        raise ValueError('workload must contain a nonempty requests list')
    identifiers = set()
    previous_arrival = -1.0
    for row in rows:
        identifier = row.get('request_id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('request_id must be a unique nonempty string')
        identifiers.add(identifier)
        arrival = row.get('arrival_s')
        if not isinstance(arrival, (int, float)) or not math.isfinite(arrival) or arrival < previous_arrival or arrival < 0:
            raise ValueError('arrival_s must be finite, nonnegative and sorted')
        previous_arrival = arrival
        tokens = row.get('prompt_token_ids')
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('prompt_token_ids must contain nonnegative integers and cannot be empty')
        maximum = row.get('max_tokens')
        if type(maximum) is not int or maximum < 1:
            raise ValueError('max_tokens must be a positive integer')
        if len(tokens) + maximum > args.max_model_len:
            raise ValueError(f'{identifier}: input + output exceeds --max-model-len')
        row.setdefault('cohort', 'custom')
        if not isinstance(row['cohort'], str):
            raise ValueError('cohort must be a string')
    return workload


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def jsonl_write(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def environment_metadata(args):
    versions = {}
    for package in ['torch', 'triton', 'transformers', 'flash-attn', 'xxhash']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    model = Path(args.model).expanduser().resolve()
    config_file = model / 'config.json'
    source_root = Path(__file__).resolve().parent
    sources = ['bench_record.py', 'record_metrics.py', 'nanovllm/engine/scheduler.py',
               'nanovllm/engine/model_runner.py', 'nanovllm/engine/block_manager.py']
    return {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(), 'packages': versions,
            'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            'model_path': str(model),
            'model_config': json.loads(config_file.read_text(encoding='utf-8')) if config_file.is_file() else None,
            'model_config_sha256': hashlib.sha256(config_file.read_bytes()).hexdigest() if config_file.is_file() else None,
            'source_sha256': {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in sources},
            'git_head': command_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD']),
            'git_status': command_output(['git', '-C', str(Path(__file__).parent), 'status', '--short']),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'nvidia_smi': command_output(['nvidia-smi']),
            'gpu_topology': command_output(['nvidia-smi', 'topo', '-m']),
            'measurement': 'host-visible engine token timestamps; no HTTP/network/tokenizer latency',
            'throughput_window': 'replay origin t=0 through final completion, including initial idle, injection gaps and drain',
            'step_timing': 'host wall clock; model_s includes GPU completion wait, not pure kernel time',
            'queue_sampling': 'per-step snapshots, not time-weighted occupancy',
            'model_revision_note': 'local config/path recorded; weight revision must additionally be pinned when downloading',
            'output_storage': 'events buffered in memory; written after the timed loop',
            'gpu_sampling': 'nvidia-smi physical GPU indices; may include GPUs outside CUDA_VISIBLE_DEVICES'}


class GPUSampler:
    def __init__(self, interval, origin, clock=time.perf_counter):
        self.interval, self.origin, self.clock = interval, origin, clock
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.interval:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        fields = ['index', 'uuid', 'name', 'utilization.gpu', 'memory.used', 'memory.total',
                  'power.draw', 'temperature.gpu', 'clocks.current.sm']
        while not self.stop_event.is_set():
            start = self.clock() - self.origin
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields),
                                         '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip())
                for row in csv.reader(result.stdout.splitlines()):
                    self.samples.append({'query_start_s': start, 'query_end_s': self.clock() - self.origin,
                                         **dict(zip(fields, (value.strip() for value in row)))})
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.samples.append({'query_start_s': start, 'error': str(exc)})
                break
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)


class Recorder:
    """Observe the existing engine; hooks are restored even if a step raises."""

    def __init__(self, workload, clock=time.perf_counter):
        self.clock = clock
        self.origin = None
        self.requests = [{
            'request_id': row['request_id'], 'cohort': row['cohort'], 'arrival_s': row['arrival_s'],
            'input_tokens': len(row['prompt_token_ids']), 'target_output_tokens': row['max_tokens'],
            'enqueued_s': None, 'first_scheduled_s': None, 'token_times_s': [], 'finish_s': None,
            'status': 'pending', 'preemptions': 0, 'cache_hit_tokens': 0,
            'prefix_cache_eligible_tokens': 0, 'recomputed_tokens': 0,
        } for row in workload['requests']]
        self.events, self.steps = [], []
        self.by_seq, self.high_water = {}, {}
        self.admitting = None
        self.active = None
        self.scheduled = []

    def now(self):
        return self.clock() - self.origin

    @staticmethod
    def state(scheduler):
        manager = scheduler.block_manager
        return {'waiting': len(scheduler.waiting), 'running': len(scheduler.running),
                'kv_free': len(manager.free_block_ids), 'kv_total': len(manager.blocks)}

    @contextmanager
    def attach(self, engine):
        scheduler = engine.scheduler
        saved = []

        def patch(owner, name, make_wrapper):
            original = getattr(owner, name)
            saved.append((owner, name, name in vars(owner), vars(owner).get(name)))
            setattr(owner, name, make_wrapper(original))

        def wrap_add(original):
            def add(seq):
                original(seq)
                record = self.admitting
                if record is None:
                    raise RuntimeError('request admission was not registered by the replay driver')
                record['seq_id'], record['enqueued_s'] = seq.seq_id, self.now()
                self.by_seq[seq.seq_id] = record
                self.high_water[seq.seq_id] = 0
                self.events.append({'event': 'enqueue', 'request_id': record['request_id'],
                                    'time_s': record['enqueued_s'], 'arrival_s': record['arrival_s']})
            return add

        def wrap_allocate(original):
            def allocate(seq, cached_blocks):
                # can_allocate excludes the final block so a final position is
                # recomputed for sampling even for a block-aligned prompt.
                eligible = ((seq.num_tokens - 1) // scheduler.block_size) * scheduler.block_size
                result = original(seq, cached_blocks)
                self.by_seq[seq.seq_id]['cache_hit_tokens'] += seq.num_cached_tokens
                self.by_seq[seq.seq_id]['prefix_cache_eligible_tokens'] += eligible
                return result
            return allocate

        def wrap_preempt(original):
            def preempt(seq):
                record = self.by_seq[seq.seq_id]
                record['preemptions'] += 1
                self.events.append({'event': 'preempt', 'request_id': record['request_id'],
                                    'time_s': self.now(), 'cached_tokens_before': seq.num_cached_tokens})
                return original(seq)
            return preempt

        def wrap_schedule(original):
            def schedule():
                started = self.now()
                try:
                    result = original()
                finally:
                    self.active['schedule_s'] = self.now() - started
                seqs, is_prefill = result
                self.active['phase'] = 'prefill' if is_prefill else 'decode'
                self.active['scheduled_tokens'] = sum(seq.num_scheduled_tokens for seq in seqs)
                self.active['kv_free_scheduled'] = len(scheduler.block_manager.free_block_ids)
                self.active['request_ids'] = [self.by_seq[seq.seq_id]['request_id'] for seq in seqs]
                self.scheduled = [(seq, seq.num_cached_tokens, seq.num_scheduled_tokens) for seq in seqs]
                for seq in seqs:
                    record = self.by_seq[seq.seq_id]
                    if record['first_scheduled_s'] is None:
                        record['first_scheduled_s'] = started
                return result
            return schedule

        def wrap_postprocess(original):
            def postprocess(seqs, token_ids, is_prefill):
                started = self.now()
                before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
                if len(token_ids) != len(seqs):
                    raise RuntimeError('unexpected result mapping: sampled token count differs from sequence count')
                try:
                    result = original(seqs, token_ids, is_prefill)
                finally:
                    self.active['postprocess_s'] = self.now() - started
                observed = self.now()
                for seq, start, count in self.scheduled:
                    record = self.by_seq[seq.seq_id]
                    end = start + count
                    record['recomputed_tokens'] += max(0, min(end, self.high_water[seq.seq_id]) - start)
                    self.high_water[seq.seq_id] = max(self.high_water[seq.seq_id], end)
                    added = seq.num_completion_tokens - before[seq.seq_id]
                    if added not in (0, 1):
                        raise RuntimeError('this recorder expects at most one new output token per request per step')
                    if added:
                        record['token_times_s'].append(observed)
                        self.active['valid_output_tokens'] += 1
                        self.events.append({'event': 'token', 'request_id': record['request_id'],
                                            'time_s': observed, 'output_index': seq.num_completion_tokens,
                                            'step_id': self.active['step_id']})
                    if seq.is_finished:
                        if seq.num_completion_tokens != record['target_output_tokens']:
                            raise RuntimeError('fixed-length output count mismatch')
                        record['status'], record['finish_s'] = 'completed', observed
                        self.events.append({'event': 'finish', 'request_id': record['request_id'], 'time_s': observed})
                return result
            return postprocess

        def wrap_run(original):
            def run(*args, **kwargs):
                started = self.now()
                try:
                    return original(*args, **kwargs)
                finally:
                    self.active['model_s'] = self.now() - started
            return run

        try:
            patch(scheduler, 'add', wrap_add)
            patch(scheduler.block_manager, 'allocate', wrap_allocate)
            patch(scheduler, 'preempt', wrap_preempt)
            patch(scheduler, 'schedule', wrap_schedule)
            patch(scheduler, 'postprocess', wrap_postprocess)
            patch(engine.model_runner, 'run', wrap_run)
            yield
        finally:
            for owner, name, existed, value in reversed(saved):
                if existed:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)

    def step(self, engine):
        before = self.state(engine.scheduler)
        self.active = {'step_id': len(self.steps), 'start_s': self.now(),
                       'schedule_s': None, 'model_s': None, 'postprocess_s': None,
                       'phase': 'unknown', 'scheduled_tokens': 0, 'valid_output_tokens': 0,
                       'prefill_token_budget': engine.scheduler.max_num_batched_tokens,
                       **{key + '_before': value for key, value in before.items() if key != 'kv_total'},
                       'kv_total': before['kv_total']}
        try:
            engine.step()
        except BaseException:
            self.active['failed'] = True
            raise
        finally:
            self.active['end_s'] = self.now()
            after = self.state(engine.scheduler)
            self.active.update({key + '_after': value for key, value in after.items() if key != 'kv_total'})
            self.steps.append(self.active)
            self.active = None


def replay(engine, workload, recorder, sampling_factory, max_seconds,
           clock=time.perf_counter, sleep=time.sleep):
    index = 0
    rows = workload['requests']
    with recorder.attach(engine):
        while index < len(rows) or not engine.is_finished():
            elapsed = clock() - recorder.origin
            if elapsed >= max_seconds:
                raise TimeoutError('replay exceeded --max-seconds between engine steps')
            while index < len(rows) and rows[index]['arrival_s'] <= clock() - recorder.origin:
                row = rows[index]
                recorder.admitting = recorder.requests[index]
                try:
                    engine.add_request(row['prompt_token_ids'], sampling_factory(row['max_tokens']))
                finally:
                    recorder.admitting = None
                index += 1
            if not engine.is_finished():
                recorder.step(engine)
            elif index < len(rows):
                sleep(max(0.0, min(0.01, rows[index]['arrival_s'] - (clock() - recorder.origin),
                                   max_seconds - (clock() - recorder.origin))))


def validate_gpu_model(args, workload, torch):
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        raise RuntimeError('run the actual benchmark on a Linux CUDA server; use --dry-run locally')
    if torch.cuda.device_count() < args.tp:
        raise ValueError('fewer visible CUDA devices than --tp')
    config = json.loads((Path(args.model).expanduser() / 'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'qwen3':
        raise ValueError('this repository runner implements Qwen3')
    for field in ['num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'vocab_size']:
        if not isinstance(config.get(field), int) or config[field] % args.tp:
            raise ValueError(f'model {field} must be divisible by --tp in this implementation')
    actual_limit = min(args.max_model_len, config['max_position_embeddings'])
    for row in workload['requests']:
        if len(row['prompt_token_ids']) + row['max_tokens'] > actual_limit:
            raise ValueError('workload exceeds model context capacity')
        if max(row['prompt_token_ids']) >= config['vocab_size']:
            raise ValueError('workload contains a token outside this model vocabulary')


def main(argv=None):
    args = parse_args(argv)
    workload = make_workload(args)
    output = args.output or Path('benchmark_records') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / 'workload.json', workload)
    print(f'Results: {output}', flush=True)
    if args.dry_run:
        print(f'Prepared {len(workload["requests"])} requests. GPU/model execution: NOT RUN.', flush=True)
        return 0
    metadata = environment_metadata(args)
    recorder = Recorder(workload)
    engine = sampler = None
    initialization_children = None
    error = None
    duration = 0.0
    try:
        import torch
        validate_gpu_model(args, workload, torch)
        from nanovllm import LLM, SamplingParams
        torch.manual_seed(args.seed)
        metadata['torch_cuda'] = torch.version.cuda
        metadata['visible_gpus'] = [{'logical_index': index, 'name': torch.cuda.get_device_name(index),
                                     'total_memory_bytes': torch.cuda.get_device_properties(index).total_memory}
                                    for index in range(args.tp)]
        started = time.perf_counter()
        initialization_children = {child.pid for child in mp.active_children()}
        engine = LLM(str(Path(args.model).expanduser().resolve()), tensor_parallel_size=args.tp,
                     enforce_eager=not args.cuda_graph, max_model_len=args.max_model_len,
                     max_num_batched_tokens=args.token_budget, max_num_seqs=args.max_num_seqs,
                     gpu_memory_utilization=args.gpu_memory_utilization)
        metadata['initialization_s'] = time.perf_counter() - started
        capacity = len(engine.scheduler.block_manager.blocks)
        if any(math.ceil((len(row['prompt_token_ids']) + row['max_tokens'] - 1) / 256) > capacity
               for row in workload['requests']):
            raise ValueError('one request alone exceeds the KV pool during generation; reduce length or increase capacity')
        make_sampling = lambda count: SamplingParams(temperature=0.6, max_tokens=count, ignore_eos=True)
        warm_rng = random.Random(args.seed + 2000003)
        started = time.perf_counter()
        for _ in range(args.warmup_runs):
            warm_rows = workload['requests'][:args.max_num_seqs]
            warm_prompts = [[warm_rng.randint(100, min(10000, metadata['model_config']['vocab_size'] - 1))
                             for _ in row['prompt_token_ids']] for row in warm_rows]
            engine.generate(warm_prompts, [make_sampling(row['max_tokens']) for row in warm_rows], use_tqdm=False)
        torch.cuda.synchronize()
        metadata['warmup_s'] = time.perf_counter() - started
        metadata['unified_kv_blocks_per_rank'] = capacity
        metadata['kv_capacity_note'] = 'local runner now takes the minimum block capacity across TP ranks before allocation'
        metadata['warmup_note'] = 'separate synthetic inputs; not a guarantee that all runtime shapes were compiled'
        recorder.origin = time.perf_counter()
        sampler = GPUSampler(args.gpu_sample_interval, recorder.origin)
        sampler.start()
        replay(engine, workload, recorder, make_sampling, args.max_seconds)
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        for record in recorder.requests:
            if record['enqueued_s'] is not None and record['status'] != 'completed':
                record['status'] = 'failed'
        print(f'Run failed: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
    finally:
        if recorder.origin is not None:
            duration = time.perf_counter() - recorder.origin
        if sampler:
            sampler.stop()
        metadata['error'] = error
        summary = summarize(recorder.requests, recorder.steps, duration,
                            slo_ttft_ms=args.slo_ttft_ms, slo_itl_ms=args.slo_itl_ms)
        summary['run_ok'] = error is None
        summary['resource_efficiency'] = {
            'allocated_gpu_seconds_per_output_token': args.tp * duration / summary['counts']['output_tokens']
            if summary['counts']['output_tokens'] else None,
            'note': 'timed workload only; excludes initialization/warmup and idle rented GPUs; not a rental bill'}
        json_write(output / 'environment.json', metadata)
        json_write(output / 'summary.json', summary)
        jsonl_write(output / 'requests.jsonl', recorder.requests)
        jsonl_write(output / 'events.jsonl', recorder.events)
        jsonl_write(output / 'steps.jsonl', recorder.steps)
        jsonl_write(output / 'gpu_samples.jsonl', sampler.samples if sampler else [])
        if engine is not None:
            atexit.unregister(engine.exit)
            try:
                engine.exit()
            except Exception as exc:
                metadata['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                error = error or {'type': 'CleanupError', 'message': str(exc)}
        elif initialization_children is not None:
            # Constructor failure can occur after TP workers were spawned. Only
            # terminate children created by this initialization, never other jobs.
            for child in mp.active_children():
                if child.pid not in initialization_children:
                    child.terminate()
                    child.join(timeout=5)
                    if child.is_alive():
                        child.kill()
                        child.join(timeout=2)
        metadata['error'] = error
        summary['run_ok'] = error is None
        json_write(output / 'environment.json', metadata)
        json_write(output / 'summary.json', summary)
    print(json.dumps({'run_ok': error is None, 'counts': summary['counts'],
                      'throughput': summary['throughput'], 'latency_ms': summary['latency_ms']},
                     ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
