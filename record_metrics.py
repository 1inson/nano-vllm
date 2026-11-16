"""Standard-library summaries for bench_record's engine-visible event records.

All timestamps and durations supplied to summarize() are in seconds. Output
latencies are in milliseconds. This module imports no model or GPU packages.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def percentiles(values: Iterable[float]) -> dict:
    """Return linear-interpolated percentiles; an empty distribution uses nulls."""
    ordered = sorted(_finite(value, "distribution value") for value in values)
    count = len(ordered)
    if not count:
        return dict(count=0, p50=None, p95=None, p99=None,
                    min=None, max=None, mean=None)

    def quantile(fraction: float) -> float:
        position = (count - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return dict(count=count, p50=quantile(0.50), p95=quantile(0.95),
                p99=quantile(0.99), min=ordered[0], max=ordered[-1],
                mean=math.fsum(ordered) / count)


def _delta_ms(later: float, earlier: float, label: str) -> float:
    difference = _finite(later, label) - _finite(earlier, label)
    if difference < -1e-9:
        raise ValueError(f"{label}: timestamps are out of order")
    return max(0.0, difference) * 1000.0


def _request_summary(requests: list[dict]) -> dict:
    samples = {key: [] for key in (
        "ttft_from_arrival", "ttft_from_enqueue", "injection_delay",
        "first_schedule_wait", "e2e_from_arrival", "itl_pooled",
        "tpot_per_request",
    )}
    completed = [row for row in requests if row.get("status") == "completed"]
    enqueued = [row for row in requests if row.get("enqueued_s") is not None]
    for row in requests:
        arrival = row["arrival_s"]
        enqueue = row.get("enqueued_s")
        first_schedule = row.get("first_scheduled_s")
        times = row.get("token_times_s", [])
        if enqueue is not None:
            samples["injection_delay"].append(
                _delta_ms(enqueue, arrival, "injection_delay"))
            if first_schedule is not None:
                samples["first_schedule_wait"].append(
                    _delta_ms(first_schedule, enqueue, "first_schedule_wait"))
        if times:
            samples["ttft_from_arrival"].append(
                _delta_ms(times[0], arrival, "ttft_from_arrival"))
            if enqueue is not None:
                samples["ttft_from_enqueue"].append(
                    _delta_ms(times[0], enqueue, "ttft_from_enqueue"))
            samples["itl_pooled"].extend(
                _delta_ms(later, earlier, "itl_pooled")
                for earlier, later in zip(times, times[1:]))
        if row.get("status") == "completed":
            if row.get("finish_s") is not None:
                samples["e2e_from_arrival"].append(
                    _delta_ms(row["finish_s"], arrival, "e2e_from_arrival"))
            if len(times) > 1:
                samples["tpot_per_request"].append(
                    _delta_ms(times[-1], times[0], "tpot_per_request") / (len(times) - 1))

    return {
        "counts": {
            "planned": len(requests),
            "enqueued": len(enqueued),
            "completed": len(completed),
            "failed": sum(row.get("status") == "failed" for row in requests),
            # Includes failed as well as still-pending requests.
            "unfinished": len(requests) - len(completed),
            "output_tokens": sum(len(row.get("token_times_s", [])) for row in requests),
            "prompt_tokens_planned": sum(row.get("input_tokens", 0) for row in requests),
            "prompt_tokens_enqueued": sum(row.get("input_tokens", 0) for row in enqueued),
            "prompt_tokens_completed": sum(row.get("input_tokens", 0) for row in completed),
        },
        "latency_ms": {key: percentiles(values) for key, values in samples.items()},
        "request_cost_totals": {
            key: sum(row.get(key, 0) for row in requests)
            for key in ("preemptions", "cache_hit_tokens", "prefix_cache_eligible_tokens",
                        "recomputed_tokens")
        },
    }


def _step_summary(steps: list[dict]) -> dict:
    grouped = defaultdict(list)
    for step in steps:
        grouped[step["phase"]].append(step)
    by_phase = {}
    for phase, rows in sorted(grouped.items()):
        duration_ms = [_delta_ms(row["end_s"], row["start_s"], "step duration")
                       for row in rows]
        result = {
            "count": len(rows),
            "duration_ms": percentiles(duration_ms),
            "total_duration_s": math.fsum(duration_ms) / 1000.0,
            "scheduled_tokens": sum(row.get("scheduled_tokens", 0) for row in rows),
            "valid_output_tokens": sum(row.get("valid_output_tokens", 0) for row in rows),
            "scheduled_tokens_per_step": percentiles(
                row.get("scheduled_tokens", 0) for row in rows),
        }
        for field in ("schedule", "model", "postprocess"):
            result[f"{field}_ms"] = percentiles(
                _finite(row[f"{field}_s"], f"{field}_s") * 1000.0
                for row in rows if row.get(f"{field}_s") is not None)
        if phase == "prefill":
            eligible = [row for row in rows if row.get("prefill_token_budget", 0) > 0]
            result["token_budget_utilization"] = percentiles(
                row.get("scheduled_tokens", 0) / row["prefill_token_budget"]
                for row in eligible)
            budget_total = sum(row["prefill_token_budget"] for row in eligible)
            result["weighted_token_budget_utilization"] = (
                sum(row.get("scheduled_tokens", 0) for row in eligible) / budget_total
                if budget_total else None)
        by_phase[phase] = result
    return {"count": len(steps), "by_phase": by_phase}


def _slo_summary(requests: list[dict], duration_s: float,
                 slo_ttft_ms: float | None, slo_itl_ms: float | None) -> dict | None:
    if slo_ttft_ms is None and slo_itl_ms is None:
        return None
    limits = {"ttft_limit_ms": slo_ttft_ms, "max_itl_limit_ms": slo_itl_ms}
    for key, value in limits.items():
        if value is not None:
            limits[key] = _finite(value, key)
            if limits[key] <= 0:
                raise ValueError(f"{key} must be positive")

    def within(value: float, limit: float) -> bool:
        return value <= limit or math.isclose(value, limit, rel_tol=1e-12, abs_tol=1e-9)

    passing = 0
    for row in requests:
        times = row.get("token_times_s", [])
        if row.get("status") != "completed" or not times:
            continue
        if limits["ttft_limit_ms"] is not None:
            ttft = _delta_ms(times[0], row["arrival_s"], "SLO TTFT")
            if not within(ttft, limits["ttft_limit_ms"]):
                continue
        if limits["max_itl_limit_ms"] is not None:
            intervals = [_delta_ms(later, earlier, "SLO ITL")
                         for earlier, later in zip(times, times[1:])]
            # A one-token response has no inter-token gap and passes this constraint.
            if intervals and not within(max(intervals), limits["max_itl_limit_ms"]):
                continue
        passing += 1
    return {
        **limits,
        "completed_requests_passing": passing,
        "fraction_of_planned": passing / len(requests) if requests else None,
        "goodput_requests_per_s": passing / duration_s if duration_s else None,
        "rule": "completed, at least one output, and all specified limits pass; "
                "ITL limit is maximum observed gap per request",
    }


def summarize(requests: list[dict], steps: list[dict], duration_s: float, *,
              slo_ttft_ms: float | None = None,
              slo_itl_ms: float | None = None) -> dict:
    """Summarize recorded engine events without model, CUDA, or network access.

    Throughput includes injection gaps and draining in its denominator, and output
    tokens include those emitted by unfinished requests. TTFT/ITL include any
    observed output; E2E and TPOT use completed requests only. Unobserved first
    tokens and unfinished responses remain visible in counts and never pass SLO.
    """
    duration_s = _finite(duration_s, "duration_s")
    if duration_s < 0:
        raise ValueError("duration_s must be nonnegative")
    request_summary = _request_summary(requests)
    counts = request_summary["counts"]
    costs = request_summary["request_cost_totals"]
    eligible_prefix_tokens = costs["prefix_cache_eligible_tokens"]
    grouped = defaultdict(list)
    for row in requests:
        grouped[str(row.get("cohort", "unknown"))].append(row)
    free_keys = ("kv_free_before", "kv_free_scheduled", "kv_free_after")
    free_blocks = [row[key] for row in steps
                   for key in free_keys
                   if row.get(key) is not None]
    capacities = [row["kv_total"] for row in steps if row.get("kv_total") is not None]
    used_blocks = [row["kv_total"] - row[key] for row in steps
                   for key in free_keys
                   if row.get("kv_total") is not None and row.get(key) is not None]
    active_requests = [row[key] for row in steps
                       for key in ("active_before", "active_scheduled", "active_after")
                       if row.get(key) is not None]
    return {
        "schema_version": 1,
        "measurement_window": {
            "duration_s": duration_s,
            "denominator": "wall time including request injection gaps and final drain",
            "latency_population": "TTFT/ITL include observed output from any status; "
                                  "E2E/TPOT use completed requests only",
            "queue_sampling": "per-step before/after snapshot distributions; "
                              "not time-weighted queue lengths",
        },
        **request_summary,
        "throughput": {
            "completed_requests_per_s": counts["completed"] / duration_s if duration_s else None,
            "output_tokens_per_s": counts["output_tokens"] / duration_s if duration_s else None,
        },
        "by_cohort": {key: _request_summary(rows) for key, rows in sorted(grouped.items())},
        "steps": _step_summary(steps),
        "queues": {
            key: percentiles(row[key] for row in steps if row.get(key) is not None)
            for key in ("waiting_before", "waiting_after", "running_before", "running_after")
        },
        "admission": {
            "peak_active_requests": max(active_requests) if active_requests else None,
            "active_request_definition": "requests holding KV, including partial prefill in waiting",
            "sampling": "before scheduling, after scheduling, and after execution snapshots",
        },
        "cache": {
            "minimum_free_blocks": min(free_blocks) if free_blocks else None,
            "maximum_used_blocks": max(used_blocks) if used_blocks else None,
            "capacity_blocks_min": min(capacities) if capacities else None,
            "capacity_blocks_max": max(capacities) if capacities else None,
            "block_sampling": "before scheduling, after scheduling, and after execution snapshots",
            "prefix_cache_hit_token_fraction": (
                costs["cache_hit_tokens"] / eligible_prefix_tokens
                if eligible_prefix_tokens > 0 else None),
            "prefix_cache_eligible_token_definition": (
                "sum of queryable complete prefix-block tokens at each allocate call, "
                "including re-allocation after preemption: "
                "((seq.num_tokens - 1) // block_size) * block_size; "
                "not the original prompt-token count"),
        },
        "slo": _slo_summary(requests, duration_s, slo_ttft_ms, slo_itl_ms),
    }
