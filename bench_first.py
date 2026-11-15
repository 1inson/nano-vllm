"""入门离线基准：固定长度、预热后测量整批耗时、输出吞吐和 PyTorch 显存。

在云端仓库目录运行：python3 -u bench_first.py --batch-size 1
再用 --batch-size 8 运行另一个进程，对比整批吞吐。
加 --cuda-graph 可启用上游已有的 CUDA Graph，其他参数相同时进行对照。
加 --profile 会在保存正式成绩后，另跑一轮诊断并导出耗时表和时间线。
使用随机 token 测性能，不评估回答质量；不测 TTFT、ITL 或在线排队时间。
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
from time import perf_counter


INPUT_TOKENS = 128
OUTPUT_TOKENS = 128
WARMUP_RUNS = 2
SEED = 0
GIB = 1024**3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cuda-graph", action="store_true", help="启用上游已有的 CUDA Graph")
    parser.add_argument("--profile", action="store_true", help="正式测量后单独采集一轮 Profiler")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 16:
        parser.error("--batch-size 必须在 1 到 16 之间")
    if args.runs < 1:
        parser.error("--runs 必须大于 0")
    return args


def make_prompts(batch_size, seed):
    # 当前引擎 KV block 大小为 256。128-token 输入不复用完整前缀块。
    # token 范围针对 Qwen3；每轮用不同输入，且相同命令可复现同一组输入。
    rng = random.Random(seed)
    return [[rng.randint(100, 10000) for _ in range(INPUT_TOKENS)]
            for _ in range(batch_size)]


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


@contextmanager
def profile_ranges(llm, record_function):
    """只在诊断阶段给原方法添加范围标签，退出时恢复原来的属性状态。"""
    runner = llm.model_runner
    targets = [
        (llm.scheduler, "schedule", "nano.schedule"),
        (llm.scheduler, "postprocess", "nano.postprocess"),
        (runner, "prepare_prefill", "nano.prepare_prefill"),
        (runner, "prepare_decode", "nano.prepare_decode"),
        (runner, "prepare_sample", "nano.prepare_sample"),
        (runner, "run_model", "nano.run_model"),
        (runner.model, "compute_logits", "nano.compute_logits"),
        (runner.sampler, "forward", "nano.sampler"),
        (runner, "run", None),
    ]
    originals = []

    def wrap(original, label):
        def call(*args, **kwargs):
            if label is None:
                is_prefill = args[1] if len(args) > 1 else kwargs["is_prefill"]
                name = "nano.prefill" if is_prefill else "nano.decode"
            else:
                name = label
            with record_function(name):
                return original(*args, **kwargs)
        return call

    try:
        for owner, name, label in targets:
            original = getattr(owner, name)
            originals.append((owner, name, name in vars(owner), vars(owner).get(name)))
            setattr(owner, name, wrap(original, label))
        yield
    finally:
        for owner, name, had_instance_value, old_value in reversed(originals):
            if had_instance_value:
                setattr(owner, name, old_value)
            else:
                delattr(owner, name)


def profile_one_batch(llm, args, sampling, result_file):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    print("开始单独的 Profiler 诊断；该轮耗时不计入正式成绩。", flush=True)
    prompts = make_prompts(args.batch_size, SEED + 100000)
    with profile_ranges(llm, record_function):
        # 范围包装可能影响编译器守卫；先运行一次，减少首次变化干扰诊断。
        llm.generate(make_prompts(args.batch_size, SEED + 99999), sampling, use_tqdm=False)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as prof:
            with record_function("nano.generate"):
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()

    lengths = [len(output["token_ids"]) for output in outputs]
    if len(lengths) != args.batch_size or any(n != OUTPUT_TOKENS for n in lengths):
        raise RuntimeError(f"诊断负载输出数量异常：{lengths}；正式成绩已经单独保存")
    trace_file = result_file.with_name(result_file.stem + "_trace.json")
    table_file = result_file.with_name(result_file.stem + "_profile.txt")
    prof.export_chrome_trace(str(trace_file))
    events = prof.key_averages()
    stages = ["阶段统计：范围内 CPU 总时长（含等待，嵌套范围不能相加）",
              f"{'阶段':<26} {'调用次数':>8} {'CPU 总计 ms':>14} {'CPU 平均 ms':>14}"]
    for event in events:
        if event.key.startswith("nano."):
            stages.append(f"{event.key:<26} {event.count:>8} "
                          f"{event.cpu_time_total / 1000:>14.3f} "
                          f"{event.cpu_time_total / event.count / 1000:>14.3f}")
    cpu_table = events.table(sort_by="self_cpu_time_total", row_limit=15)
    device_table = events.table(sort_by="self_device_time_total", row_limit=15)
    text = "\n".join(stages) + "\n\nCPU 操作表：\n" + cpu_table
    text += "\n\n设备操作表：\n" + device_table
    text += ("\n说明：Profiler 和范围标签有额外开销，不用于正式成绩。\n"
             "CPU 时间包含同步等待；不要把 .tolist()/同步时间直接解释为复制成本。\n"
             "设备操作表不保证完整归因 CUDA Graph 内部的所有 kernel，请结合 trace 时间线。\n")
    table_file.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"耗时表已保存：{table_file}", flush=True)
    print(f"时间线已保存：{trace_file}", flush=True)


def main():
    args = parse_args()
    model_dir = Path(args.model).expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"请先下载模型：{model_dir}")

    import torch
    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("需要在已配置好的云端 GPU 环境运行")
    torch.manual_seed(SEED)
    engine_config = dict(
        enforce_eager=not args.cuda_graph,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_num_seqs=16,
        gpu_memory_utilization=0.60,
    )
    sampling = SamplingParams(temperature=0.6, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    mode = "cuda_graph" if args.cuda_graph else "eager"
    print(f"执行模式：{mode}（两种模式均可能使用 torch.compile）。", flush=True)
    print(f"配置：每批 {args.batch_size} 条请求，每条输入 {INPUT_TOKENS} token，"
          f"强制输出 {OUTPUT_TOKENS} token；预热 {WARMUP_RUNS} 轮，测量 {args.runs} 轮。",
          flush=True)
    print("初始化引擎（包括内部预热），此阶段单独计时。", flush=True)
    torch.cuda.synchronize()
    start = perf_counter()
    llm = LLM(str(model_dir), **engine_config)
    torch.cuda.synchronize()
    init_seconds = perf_counter() - start
    print(f"引擎初始化：{init_seconds:.3f} 秒。开始相同规模的负载预热。", flush=True)

    for index in range(WARMUP_RUNS):
        llm.generate(make_prompts(args.batch_size, SEED + index), sampling, use_tqdm=False)
        torch.cuda.synchronize()
        print(f"预热 {index + 1}/{WARMUP_RUNS} 完成。", flush=True)

    rows = []
    for index in range(args.runs):
        prompts = make_prompts(args.batch_size, SEED + WARMUP_RUNS + index)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        seconds = perf_counter() - start

        lengths = [len(output["token_ids"]) for output in outputs]
        if len(lengths) != args.batch_size or any(n != OUTPUT_TOKENS for n in lengths):
            raise RuntimeError(f"实际输出长度不符合固定负载，停止记录：{lengths}")
        count = sum(lengths)
        row = dict(
            run=index + 1,
            batch_seconds=seconds,
            output_tokens=count,
            output_tokens_per_second=count / seconds,
            peak_torch_allocated_gib=torch.cuda.max_memory_allocated() / GIB,
            peak_torch_reserved_gib=torch.cuda.max_memory_reserved() / GIB,
        )
        rows.append(row)
        print(f"第 {index + 1} 轮 | 整批耗时 {seconds:.3f} s | 实际输出 {count} token | "
              f"输出吞吐 {row['output_tokens_per_second']:.2f} token/s | "
              f"PyTorch 已分配峰值 {row['peak_torch_allocated_gib']:.2f} GiB | "
              f"PyTorch 保留峰值 {row['peak_torch_reserved_gib']:.2f} GiB", flush=True)

    summary = dict(
        median_batch_seconds=statistics.median(row["batch_seconds"] for row in rows),
        median_output_tokens_per_second=statistics.median(
            row["output_tokens_per_second"] for row in rows),
    )
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        packages={name: importlib.metadata.version(name)
                  for name in ("torch", "triton", "transformers", "flash-attn")},
        torch_cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        nvidia_smi=command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader"]),
        git_commit=command_output(["git", "-C", str(Path(__file__).resolve().parent),
                                   "rev-parse", "HEAD"]),
        model_dir=str(model_dir),
        model_config=json.loads((model_dir / "config.json").read_text(encoding="utf-8")),
        engine=engine_config,
        mode=mode,
        workload=dict(batch_size=args.batch_size, input_tokens=INPUT_TOKENS,
                      output_tokens=OUTPUT_TOKENS, temperature=0.6, ignore_eos=True,
                      seed=SEED, warmup_runs=WARMUP_RUNS, measured_runs=args.runs),
        engine_init_seconds=init_seconds,
        measurements=rows,
        summary=summary,
        notes=[
            "Offline synthetic workload, not a text quality evaluation.",
            "Timing includes generate(): CPU scheduling, GPU compute and final text decoding.",
            "Input token IDs are prepared before timing; model loading and warmup are excluded.",
            "Whole-batch elapsed time is not TTFT, ITL, or per-request latency for a batch.",
            "PyTorch memory includes the preallocated KV pool, not just active request memory.",
            "PyTorch allocator metrics exclude some driver/NCCL allocations; they are not nvidia-smi totals.",
            "Two warmup rounds do not guarantee every compiled shape is warmed; inspect per-run stability.",
            "Model config and git commit are recorded; weight bytes and uncommitted edits are not hashed.",
        ],
    )
    output_dir = Path(__file__).resolve().parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_file = output_dir / f"{stamp}_{mode}_b{args.batch_size}.json"
    with output_file.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"中位整批耗时：{summary['median_batch_seconds']:.3f} s；"
          f"中位输出吞吐：{summary['median_output_tokens_per_second']:.2f} token/s。")
    print("显存数值包含引擎预分配的 KV 缓存池，不代表这些请求实际需要的最小显存。")
    print(f"结果已保存：{output_file}", flush=True)
    if args.profile:
        profile_one_batch(llm, args, sampling, output_file)


if __name__ == "__main__":
    main()
