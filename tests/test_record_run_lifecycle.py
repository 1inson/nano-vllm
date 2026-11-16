"""CPU-only integration checks for benchmark reporting and TP worker cleanup."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_record


@contextmanager
def temporary_results():
    # Keep all test writes inside the checkout, including under restricted hosts.
    test_directory = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="nano_lifecycle_", dir=test_directory) as directory:
        if Path(directory).resolve().parent != test_directory:
            raise RuntimeError("test temporary directory escaped the checkout")
        yield directory


class RunLifecycleTests(unittest.TestCase):
    def run_main(self, temporary_directory, *, exit_error=None, constructor_error=None,
                 active_children=None, extra_args=()):
        output = Path(temporary_directory) / "recording"
        engine = SimpleNamespace(
            scheduler=SimpleNamespace(block_manager=SimpleNamespace(blocks=list(range(16)))),
            exit=Mock(side_effect=exit_error),
        )
        constructor = Mock(return_value=engine, side_effect=constructor_error)
        nanovllm = ModuleType("nanovllm")
        nanovllm.LLM = constructor
        nanovllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        torch = ModuleType("torch")
        torch.manual_seed = Mock()
        torch.version = SimpleNamespace(cuda="mock-cuda")
        torch.cuda = SimpleNamespace(
            get_device_name=Mock(return_value="mock GPU"),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=2**30)),
            synchronize=Mock(),
        )
        sampler = SimpleNamespace(samples=[], start=Mock(), stop=Mock())

        def finish_one_request(engine_argument, workload, recorder, sampling_factory, deadline):
            self.assertIs(engine_argument, engine)
            record = recorder.requests[0]
            enqueue_time = recorder.now()
            finish_time = recorder.now()
            record.update(
                seq_id=0, enqueued_s=enqueue_time, first_scheduled_s=enqueue_time,
                token_times_s=[finish_time], finish_s=finish_time, status="completed",
            )

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"torch": torch, "nanovllm": nanovllm}))
            stack.enter_context(patch.object(bench_record, "validate_gpu_model"))
            stack.enter_context(patch.object(
                bench_record, "environment_metadata", return_value={"model_config": {"vocab_size": 10001}},
            ))
            sampler_factory = stack.enter_context(patch.object(bench_record, "GPUSampler", return_value=sampler))
            replay = stack.enter_context(patch.object(bench_record, "replay", side_effect=finish_one_request))
            unregister = stack.enter_context(patch.object(bench_record.atexit, "unregister"))
            children = stack.enter_context(patch.object(
                bench_record.mp, "active_children", side_effect=active_children or [[]],
            ))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            result = bench_record.main([
                "--output", str(output), "--requests", "1", "--workload", "short",
                "--short-input", "1", "--long-input", "1", "--output-tokens", "1",
                "--arrival-rate", "0", "--warmup-runs", "0", "--gpu-sample-interval", "0",
                "--tp", "2",
            ] + list(extra_args))

        return SimpleNamespace(
            result=result,
            summary=json.loads((output / "summary.json").read_text(encoding="utf-8")),
            metadata=json.loads((output / "environment.json").read_text(encoding="utf-8")),
            requests=[json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()],
            engine=engine, constructor=constructor, sampler=sampler,
            sampler_factory=sampler_factory, replay=replay, unregister=unregister,
            children=children,
        )

    def test_successful_run_records_success_and_cleans_up(self):
        with temporary_results() as temporary_directory:
            run = self.run_main(temporary_directory)
        self.assertEqual(run.result, 0)
        self.assertTrue(run.summary["run_ok"])
        self.assertIsNone(run.metadata["error"])
        self.assertEqual(run.summary["counts"]["completed"], 1)
        self.assertEqual(run.requests[0]["status"], "completed")
        run.engine.exit.assert_called_once_with()
        run.unregister.assert_called_once_with(run.engine.exit)
        run.sampler.start.assert_called_once_with()
        run.sampler.stop.assert_called_once_with()
        run.replay.assert_called_once()
        self.assertIsNone(run.constructor.call_args.kwargs["max_active_requests"])

    def test_admission_cap_reaches_engine_configuration(self):
        with temporary_results() as temporary_directory:
            run = self.run_main(temporary_directory, extra_args=["--max-active-requests", "16"])
        self.assertEqual(run.result, 0)
        self.assertEqual(run.constructor.call_args.kwargs["max_active_requests"], 16)

    def test_exit_failure_is_reflected_in_final_summary_and_metadata(self):
        with temporary_results() as temporary_directory:
            run = self.run_main(temporary_directory, exit_error=RuntimeError("shutdown failed"))
        self.assertEqual(run.result, 1)
        self.assertFalse(run.summary["run_ok"])
        self.assertEqual(run.metadata["error"]["type"], "CleanupError")
        self.assertEqual(run.metadata["error"]["message"], "shutdown failed")
        self.assertIn("shutdown failed", run.metadata["cleanup_error"])
        # The workload itself completed; cleanup failure must still make the run fail.
        self.assertEqual(run.summary["counts"]["completed"], 1)
        run.engine.exit.assert_called_once_with()
        run.sampler.stop.assert_called_once_with()

    def test_constructor_failure_cleans_only_new_workers(self):
        existing_worker = Mock(pid=101)
        new_worker = Mock(pid=202)
        # Exercise the bounded kill fallback after terminate did not finish it.
        new_worker.is_alive.return_value = True
        with temporary_results() as temporary_directory:
            run = self.run_main(
                temporary_directory, constructor_error=RuntimeError("initialization failed"),
                active_children=[[existing_worker], [existing_worker, new_worker]],
            )
        self.assertEqual(run.result, 1)
        self.assertFalse(run.summary["run_ok"])
        self.assertEqual(run.metadata["error"]["type"], "RuntimeError")
        self.assertEqual(run.metadata["error"]["message"], "initialization failed")
        self.assertEqual(run.children.call_count, 2)
        existing_worker.terminate.assert_not_called()
        existing_worker.join.assert_not_called()
        existing_worker.kill.assert_not_called()
        new_worker.terminate.assert_called_once_with()
        self.assertEqual(new_worker.join.call_args_list[0].kwargs, {"timeout": 5})
        new_worker.kill.assert_called_once_with()
        self.assertEqual(new_worker.join.call_args_list[1].kwargs, {"timeout": 2})
        run.engine.exit.assert_not_called()
        run.sampler_factory.assert_not_called()
        run.replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
