"""CPU-only checks of metric definitions and censored/incomplete requests."""

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from record_metrics import percentiles, summarize


def request(identifier="r0", **overrides):
    record = dict(request_id=identifier, cohort="short", arrival_s=0.0,
                  enqueued_s=0.1, first_scheduled_s=0.2,
                  token_times_s=[0.3, 0.4, 0.5], finish_s=0.51,
                  input_tokens=128, target_output_tokens=3, status="completed",
                  preemptions=0, cache_hit_tokens=0, recomputed_tokens=0)
    record.update(overrides)
    return record


def step(phase="prefill", **overrides):
    record = dict(phase=phase, start_s=0.0, end_s=0.2, schedule_s=0.01,
                  model_s=0.18, postprocess_s=0.01, scheduled_tokens=600,
                  valid_output_tokens=1, waiting_before=2, waiting_after=1,
                  running_before=0, running_after=1, kv_free_before=20,
                  kv_free_after=17, kv_total=32, prefill_token_budget=1024)
    record.update(overrides)
    return record


class PercentileTests(unittest.TestCase):
    def test_irregular_distribution_uses_linear_interpolation(self):
        result = percentiles([1, 2, 7, 10])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["p50"], 4.5)
        self.assertAlmostEqual(result["p95"], 9.55)
        self.assertAlmostEqual(result["p99"], 9.91)
        self.assertEqual(result["mean"], 5.0)

    def test_empty_singleton_and_generator(self):
        self.assertEqual(percentiles([]), dict(count=0, p50=None, p95=None,
                                              p99=None, min=None, max=None, mean=None))
        self.assertEqual(percentiles(iter([4]))["p99"], 4.0)
        self.assertEqual(percentiles(value for value in [2, 4])["mean"], 3.0)

    def test_nonfinite_is_explicit_error(self):
        for value in [math.inf, -math.inf, math.nan]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                percentiles([value])


class SummaryTests(unittest.TestCase):
    def test_empty_summary_is_json_serializable_and_avoids_zero_division(self):
        result = summarize([], [], 0, slo_ttft_ms=100)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["counts"]["planned"], 0)
        self.assertIsNone(result["throughput"]["output_tokens_per_s"])
        self.assertIsNone(result["cache"]["minimum_free_blocks"])
        self.assertIsNone(result["slo"]["fraction_of_planned"])

    def test_arrival_and_enqueue_clocks_are_not_conflated(self):
        result = summarize([request()], [], 2)
        latency = result["latency_ms"]
        self.assertAlmostEqual(latency["ttft_from_arrival"]["p50"], 300)
        self.assertAlmostEqual(latency["ttft_from_enqueue"]["p50"], 200)
        self.assertAlmostEqual(latency["injection_delay"]["p50"], 100)
        self.assertAlmostEqual(latency["first_schedule_wait"]["p50"], 100)
        self.assertAlmostEqual(latency["e2e_from_arrival"]["p50"], 510)
        self.assertEqual(result["throughput"]["completed_requests_per_s"], 0.5)
        self.assertEqual(result["throughput"]["output_tokens_per_s"], 1.5)

    def test_pooled_itl_and_request_tpot_have_different_weighting(self):
        rows = [
            request("short", token_times_s=[0.3, 0.31], finish_s=0.32),
            request("long", cohort="long", token_times_s=[0.3, 0.4, 0.5, 0.6], finish_s=0.61),
        ]
        result = summarize(rows, [], 1)
        self.assertEqual(result["latency_ms"]["itl_pooled"]["count"], 4)
        self.assertAlmostEqual(result["latency_ms"]["itl_pooled"]["mean"], 77.5)
        self.assertEqual(result["latency_ms"]["tpot_per_request"]["count"], 2)
        self.assertAlmostEqual(result["latency_ms"]["tpot_per_request"]["mean"], 55)
        self.assertEqual(result["by_cohort"]["long"]["counts"]["planned"], 1)

    def test_zero_and_single_output_do_not_create_fake_intervals(self):
        rows = [request("zero", token_times_s=[], status="failed", finish_s=None),
                request("one", token_times_s=[0.3], finish_s=0.3)]
        result = summarize(rows, [], 1, slo_ttft_ms=500, slo_itl_ms=1)
        self.assertEqual(result["latency_ms"]["itl_pooled"]["count"], 0)
        self.assertEqual(result["latency_ms"]["tpot_per_request"]["count"], 0)
        self.assertEqual(result["slo"]["completed_requests_passing"], 1)
        self.assertEqual(result["slo"]["fraction_of_planned"], 0.5)

    def test_failures_and_unsubmitted_requests_are_kept_in_denominator(self):
        rows = [request("good"),
                request("failed", status="failed", token_times_s=[0.3], finish_s=0.4),
                request("pending", status="pending", token_times_s=[0.3, 0.4], finish_s=None),
                request("future", status="pending", enqueued_s=None,
                        first_scheduled_s=None, token_times_s=[], finish_s=None)]
        result = summarize(rows, [], 2, slo_ttft_ms=400, slo_itl_ms=150)
        counts = result["counts"]
        self.assertEqual((counts["planned"], counts["enqueued"], counts["completed"],
                          counts["failed"], counts["unfinished"]), (4, 3, 1, 1, 3))
        self.assertEqual(counts["output_tokens"], 6)
        self.assertEqual(counts["prompt_tokens_planned"], 512)
        self.assertEqual(counts["prompt_tokens_enqueued"], 384)
        self.assertEqual(result["slo"]["fraction_of_planned"], 0.25)
        self.assertEqual(result["slo"]["goodput_requests_per_s"], 0.5)
        self.assertEqual(result["latency_ms"]["ttft_from_arrival"]["count"], 3)
        self.assertEqual(result["latency_ms"]["e2e_from_arrival"]["count"], 1)
        self.assertEqual(result["latency_ms"]["tpot_per_request"]["count"], 1)

    def test_slo_checks_both_constraints_and_worst_gap(self):
        rows = [request("good", token_times_s=[0.3, 0.35, 0.4]),
                request("bad_itl", token_times_s=[0.3, 0.31, 0.5]),
                request("bad_ttft", token_times_s=[0.6, 0.61])]
        result = summarize(rows, [], 1, slo_ttft_ms=400, slo_itl_ms=100)
        self.assertEqual(result["slo"]["completed_requests_passing"], 1)
        self.assertAlmostEqual(result["slo"]["fraction_of_planned"], 1 / 3)
        self.assertIsNone(summarize(rows, [], 1)["slo"])

    def test_slo_boundary_is_not_broken_by_float_roundoff(self):
        result = summarize([request(token_times_s=[0.3, 0.35, 0.4])], [], 1,
                           slo_ttft_ms=300, slo_itl_ms=50)
        self.assertEqual(result["slo"]["completed_requests_passing"], 1)

    def test_phase_budget_cache_and_cost_summaries(self):
        steps = [step(), step(scheduled_tokens=424, kv_free_before=17, kv_free_after=15,
                              prefill_token_budget=512),
                 step("decode", scheduled_tokens=2, valid_output_tokens=2,
                      kv_free_before=15, kv_free_after=32)]
        rows = [request(preemptions=2, cache_hit_tokens=256, recomputed_tokens=123)]
        result = summarize(rows, steps, 1)
        prefill = result["steps"]["by_phase"]["prefill"]
        self.assertEqual(prefill["count"], 2)
        self.assertEqual(prefill["scheduled_tokens"], 1024)
        self.assertAlmostEqual(prefill["weighted_token_budget_utilization"], 1024 / 1536)
        self.assertAlmostEqual(prefill["model_ms"]["mean"], 180)
        self.assertNotIn("token_budget_utilization", result["steps"]["by_phase"]["decode"])
        self.assertEqual(result["cache"]["minimum_free_blocks"], 15)
        self.assertEqual(result["cache"]["maximum_used_blocks"], 17)
        self.assertEqual(result["queues"]["waiting_before"]["count"], 3)
        self.assertEqual(result["request_cost_totals"],
                         dict(preemptions=2, cache_hit_tokens=256,
                              prefix_cache_eligible_tokens=0, recomputed_tokens=123))

    def test_scheduled_snapshot_captures_kv_peak_before_completion_frees_it(self):
        result = summarize([request()], [step(kv_free_before=32, kv_free_scheduled=4,
                                             kv_free_after=32)], 1)
        self.assertEqual(result["cache"]["minimum_free_blocks"], 4)
        self.assertEqual(result["cache"]["maximum_used_blocks"], 28)
        legacy = summarize([request()], [step(kv_free_before=32, kv_free_after=32)], 1)
        self.assertEqual(legacy["cache"]["minimum_free_blocks"], 32)
        self.assertEqual(legacy["cache"]["maximum_used_blocks"], 0)

    def test_prefix_hit_ratio_uses_all_allocation_queries_not_original_prompts(self):
        rows = [request("repeat", cohort="long", input_tokens=128,
                        preemptions=2, cache_hit_tokens=512, prefix_cache_eligible_tokens=1024),
                request("miss", cache_hit_tokens=0, prefix_cache_eligible_tokens=768)]
        result = summarize(rows, [], 1)
        self.assertEqual(result["request_cost_totals"]["prefix_cache_eligible_tokens"], 1792)
        self.assertAlmostEqual(result["cache"]["prefix_cache_hit_token_fraction"], 512 / 1792)
        self.assertEqual(result["by_cohort"]["long"]["request_cost_totals"]["prefix_cache_eligible_tokens"], 1024)
        self.assertIn("re-allocation", result["cache"]["prefix_cache_eligible_token_definition"])
        self.assertIsNone(summarize([request()], [], 1)["cache"]["prefix_cache_hit_token_fraction"])

    def test_queue_distributions_state_snapshot_not_time_weighted_sampling(self):
        steps = [step(waiting_before=10, end_s=1), step(waiting_before=0, end_s=100)]
        result = summarize([], steps, 100)
        self.assertEqual(result["queues"]["waiting_before"]["mean"], 5)
        self.assertIn("not time-weighted", result["measurement_window"]["queue_sampling"])

    def test_invalid_timestamps_and_duration_fail_loudly(self):
        for duration in [-1, math.inf, math.nan]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                summarize([], [], duration)
        with self.assertRaises(ValueError):
            summarize([request(token_times_s=[0.3, 0.2])], [], 1)
        with self.assertRaises(ValueError):
            summarize([], [step(start_s=1, end_s=0)], 1)
        for limit in [0, -1, math.inf]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                summarize([], [], 1, slo_ttft_ms=limit)


if __name__ == "__main__":
    unittest.main()
