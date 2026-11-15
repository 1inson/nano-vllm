"""Read-only rental checks for this single-host nano-vllm implementation.

Examples (run on the rented Linux server, using the same Python as the engine):
    python check_multi_gpu.py --tp 2 --model ~/huggingface/Qwen3-0.6B
    python check_multi_gpu.py --tp 2 --model ~/huggingface/Qwen3-0.6B --collective-test

The default checks do not create an engine or initialize a process group. The
optional test creates only its own short-lived NCCL workers. Never launch the
engine with torchrun: LLMEngine already starts its own tensor-parallel workers.
No packages or models are downloaded, and existing shared memory is never removed.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import shutil
import socket
import subprocess
import sys
import time
import traceback


def model_checks(config: dict, tp: int) -> list[tuple[str, str, str]]:
    """Check the actual sharding rules without importing torch/transformers."""
    checks = []
    if config.get("model_type") != "qwen3":
        checks.append(("FAIL", "model_type", "this runner implements dense Qwen3 only"))
    else:
        checks.append(("OK", "model_type", "qwen3"))
    fields = ("hidden_size", "num_attention_heads", "num_key_value_heads",
              "intermediate_size", "vocab_size")
    values = {}
    for name in fields:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            checks.append(("FAIL", name, "expected a positive integer in config.json"))
            continue
        values[name] = value
        if name == "hidden_size":
            # Qwen3 may specify head_dim explicitly. Hidden width is not itself
            # sharded: o_proj shards Q-heads * head_dim, MLP shards intermediate.
            checks.append(("OK", name, f"{value}; no direct hidden_size % TP requirement"))
        elif value % tp:
            checks.append(("FAIL", name, f"{value} is not divisible by TP={tp}"))
        else:
            checks.append(("OK", name, f"{value}; {value // tp} per rank"))
    q_heads = values.get("num_attention_heads")
    kv_heads = values.get("num_key_value_heads")
    if q_heads and kv_heads and q_heads % kv_heads:
        checks.append(("FAIL", "GQA heads", "Q heads must be a multiple of KV heads"))
    head_dim = config.get("head_dim")
    if head_dim is None and q_heads and "hidden_size" in values:
        hidden = values["hidden_size"]
        if hidden % q_heads:
            checks.append(("FAIL", "head_dim", "missing head_dim and hidden_size / Q heads is not integral"))
        else:
            head_dim = hidden // q_heads
    if head_dim is not None:
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            checks.append(("FAIL", "head_dim", "expected a positive integer"))
        else:
            checks.append(("OK", "head_dim", str(head_dim)))
    return checks


def can_bind(host: str, port: int) -> tuple[bool, str]:
    """A point-in-time check; does not reserve the engine port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
        return True, f"{host}:{port} available now (not reserved)"
    except OSError as exc:
        return False, f"{host}:{port}: {exc}"


def _collective_worker(rank: int, tp: int, init_method: str, timeout: float, results):
    """Symmetric operations also exercise the gather used by ParallelLMHead."""
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", init_method=init_method, rank=rank, world_size=tp,
            timeout=timedelta(seconds=timeout),
        )
        local = torch.arange(1024, dtype=torch.float32, device=f"cuda:{rank}")
        expected_local = 1023 * 1024 / 2
        sanity_ok = local.sum().item() == expected_local
        reduced = torch.full((1024,), float(rank + 1), device=f"cuda:{rank}")
        dist.all_reduce(reduced)
        reduced_ok = bool(torch.all(reduced == tp * (tp + 1) / 2).item())
        source = torch.tensor([rank], dtype=torch.int64, device=f"cuda:{rank}")
        gathered = [torch.empty_like(source) for _ in range(tp)] if rank == 0 else None
        dist.gather(source, gather_list=gathered, dst=0)
        gather_ok = rank != 0 or [int(item.item()) for item in gathered] == list(range(tp))
        # Do not raise before all peers finish their symmetric collectives.
        torch.cuda.synchronize(rank)
        dist.barrier()
        results.put((rank, sanity_ok and reduced_ok and gather_ok,
                     "CUDA sanity + NCCL all_reduce + gather"))
    except Exception:
        results.put((rank, False, traceback.format_exc(limit=3)))
    finally:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def run_collective_test(tp: int, timeout: float) -> tuple[bool, list[str]]:
    """Bound runtime and clean up only the children started by this function."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # The port is released before spawn; another process can theoretically race.
    context = mp.get_context("spawn")
    results = context.Queue()
    processes = []
    reports = {}
    messages = []
    deadline = time.monotonic() + timeout
    try:
        for rank in range(tp):
            worker = context.Process(
                target=_collective_worker,
                args=(rank, tp, f"tcp://127.0.0.1:{port}", timeout, results),
            )
            worker.start()
            processes.append(worker)
        while time.monotonic() < deadline and len(reports) < tp:
            try:
                rank, ok, detail = results.get(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
                reports[rank] = (ok, detail)
                if not ok:
                    break
            except queue.Empty:
                # A hard-crashed worker may be unable to put an error result.
                if any(p.exitcode not in (None, 0) for p in processes):
                    break
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for rank in range(tp):
            if rank in reports:
                ok, detail = reports[rank]
                messages.append(f"rank {rank}: {'OK' if ok else 'FAIL'} {detail}")
            else:
                messages.append(f"rank {rank}: FAIL no result (timeout or worker failure)")
        success = (len(reports) == tp and all(ok for ok, _ in reports.values())
                   and all(p.exitcode == 0 for p in processes))
        if not success:
            messages.append(f"worker exit codes: {[p.exitcode for p in processes]}")
        return success, messages
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        results.close()
        results.cancel_join_thread()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--model", type=Path, help="local model directory (config.json checked without loading weights)")
    parser.add_argument("--collective-test", action="store_true", help="explicitly start temporary NCCL workers")
    parser.add_argument("--timeout", type=float, default=60, help="collective-test total timeout in seconds (default: 60)")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    return args


def main(argv=None):
    args = parse_args(argv)
    checks = []

    def emit(level, name, detail):
        checks.append((level, name, detail))
        print(f"[{level}] {name}: {detail}", flush=True)

    linux = platform.system() == "Linux"
    emit("OK" if linux else "FAIL", "OS", f"{platform.system()} {platform.release()}; engine requires Linux/NCCL")
    emit("OK" if (3, 10) <= sys.version_info[:2] < (3, 13) else "FAIL", "Python", sys.version.split()[0])

    available, detail = can_bind("127.0.0.1", 2333)
    emit("OK" if available else "FAIL", "fixed engine port", detail)
    if linux:
        shm_path = Path("/dev/shm")
        try:
            free = shutil.disk_usage(shm_path).free
            emit("OK" if free >= 2**20 else "FAIL" if args.tp > 1 else "WARN", "shared memory",
                 f"{free / 2**20:.1f} MiB free; TP>1 uses a fixed 1 MiB segment")
            exists = (shm_path / "nanovllm").exists()
            emit("FAIL" if exists and args.tp > 1 else "WARN" if exists else "OK",
                 "fixed shared-memory name", "nanovllm already exists; inspect its owner, do not delete a live engine's segment" if exists else "nanovllm not present")
        except OSError as exc:
            emit("FAIL" if args.tp > 1 else "WARN", "/dev/shm", str(exc))
    else:
        emit("WARN", "/dev/shm", "Linux-only check skipped")

    if args.model:
        model = args.model.expanduser().resolve()
        try:
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            for level, name, detail in model_checks(config, args.tp):
                emit(level, f"model {name}", detail)
            emit("OK" if any(model.glob("*.safetensors")) else "FAIL", "weight files",
                 "safetensors files found; not loaded or fully verified" if any(model.glob("*.safetensors")) else "no safetensors files found")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            emit("FAIL", "model config", f"{type(exc).__name__}: {exc}")
    else:
        emit("WARN", "model config", "pass --model to check TP divisibility and local weight presence")

    cuda_ready = False
    try:
        import torch
        import torch.distributed as dist
        emit("OK", "PyTorch", f"{torch.__version__}; compiled CUDA={torch.version.cuda}")
        nccl = dist.is_available() and dist.is_nccl_available()
        emit("OK" if nccl else "FAIL", "NCCL backend", "available (not initialized)" if nccl else "unavailable")
        count = torch.cuda.device_count()
        emit("OK" if count >= args.tp else "FAIL", "visible GPUs", f"{count}; requested TP={args.tp}, engine uses logical devices 0..{args.tp - 1}")
        selected = []
        for index in range(min(count, args.tp)):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            selected.append((props.name, total, free))
            capability = (props.major, props.minor)
            emit("OK" if props.major >= 8 else "FAIL", f"GPU {index}",
                 f"{props.name}; capacity={total / 2**30:.2f} GiB; free={free / 2**30:.2f} GiB; compute={capability[0]}.{capability[1]} (FlashAttention-2 needs Ampere or newer)")
        if len({(name, total) for name, total, _ in selected}) > 1:
            emit("WARN", "mixed GPU models/capacities", "prefer identical GPUs; the shared KV block count is limited by the smallest rank and performance by slow ranks")
        if len(selected) > 1 and max(free for _, _, free in selected) - min(free for _, _, free in selected) > 2**30:
            emit("WARN", "unequal free GPU memory", "more than 1 GiB difference; check co-tenants and per-rank KV capacity")
        cuda_ready = linux and nccl and count >= args.tp and all(
            torch.cuda.get_device_properties(index).major >= 8 for index in range(args.tp)
        )
    except (ImportError, OSError, RuntimeError) as exc:
        emit("FAIL", "CUDA inspection", f"{type(exc).__name__}: {exc}")

    try:
        result = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            emit("OK", "physical GPU topology", "physical GPU numbering may differ from CUDA_VISIBLE_DEVICES logical ranks")
            print(result.stdout.strip(), flush=True)
        else:
            emit("WARN", "GPU topology", result.stderr.strip() or f"nvidia-smi exited {result.returncode}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        emit("WARN", "GPU topology", str(exc))

    if args.collective_test:
        if cuda_ready:
            emit("INFO", "collective test", "starting only this check's NCCL workers on a temporary localhost port")
            success, details = run_collective_test(args.tp, args.timeout)
            for detail in details:
                print(detail, flush=True)
            emit("OK" if success else "FAIL", "collective test", "passed" if success else "failed; its workers were cleaned up")
        else:
            emit("FAIL", "collective test", "requires Linux, enough compatible visible GPUs and an NCCL-enabled PyTorch")
    else:
        emit("INFO", "collective test", "not run; add --collective-test before benchmarking a rented multi-GPU instance")

    failures = sum(level == "FAIL" for level, _, _ in checks)
    warnings = sum(level == "WARN" for level, _, _ in checks)
    print(f"\nResult: {failures} failed checks, {warnings} warnings. This is a preflight, not a model correctness or scaling benchmark.", flush=True)
    print("Launch one Python engine process; it spawns its TP workers. Run one engine per host with this fixed-port/shared-memory implementation.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
