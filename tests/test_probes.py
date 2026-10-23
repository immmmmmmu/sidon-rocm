"""Portable regression checks; fixtures are not Windows/GPU acceptance evidence."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sidon_rocm_windows import cli


def torch_fixture(hip="7.2", available=True):
    return SimpleNamespace(
        __version__="2.9.1+rocm7.2.1",
        version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index: "AMD Radeon(TM) 8060S Graphics",
            get_device_properties=lambda index: SimpleNamespace(total_memory=1234, gcnArchName="gfx1151"),
        ),
    )


class ProbeTests(unittest.TestCase):
    def inspect_fixture(self, torch):
        with patch.object(cli.platform, "system", return_value="Windows"), \
             patch.object(cli.platform, "release", return_value="11"), \
             patch.object(cli.sys, "version_info", (3, 12, 0)), \
             patch.object(cli.importlib, "import_module", return_value=torch):
            return cli.environment_report()[0]

    def test_cuda_available_without_hip_is_blocked(self):
        report = self.inspect_fixture(torch_fixture(hip=None))
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["hip_backend"])

    def test_missing_gpu_is_blocked(self):
        report = self.inspect_fixture(torch_fixture(available=False))
        self.assertEqual(report["status"], "blocked")
        self.assertNotIn("gpu", report)

    def test_wrong_version_and_architecture_are_blocked(self):
        torch = torch_fixture()
        torch.__version__ = "2.9.1+cpu"
        self.assertEqual(self.inspect_fixture(torch)["status"], "blocked")
        torch = torch_fixture()
        torch.cuda.get_device_properties = lambda index: SimpleNamespace(total_memory=1234, gcnArchName="gfx1201")
        self.assertEqual(self.inspect_fixture(torch)["status"], "blocked")

    def test_fixture_only_environment_can_pass(self):
        self.assertEqual(self.inspect_fixture(torch_fixture())["status"], "passed")

    def test_torch_import_failure_is_reported_without_private_trace(self):
        with patch.object(cli.importlib, "import_module", side_effect=ImportError("private path")):
            report, torch = cli.environment_report()
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(torch)
        self.assertEqual(report["error"]["type"], "ImportError")
        self.assertNotIn("private path", json.dumps(report))

    def test_blocked_environment_does_not_compute(self):
        report = cli.smoke_report({"status": "blocked"}, None, "float32")
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["reason"], "environment_gate_failed")

    def test_computation_exception_is_failure(self):
        torch = SimpleNamespace(float32="fp32", float16="fp16", inference_mode=contextlib.nullcontext,
                                cuda=SimpleNamespace(reset_peak_memory_stats=lambda index: None))
        report = cli.smoke_report({"status": "passed"}, torch, "float32")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["reason"], "gpu_computation_failed")

    def test_incorrect_nonfinite_or_cpu_result_is_failure(self):
        for device, finite, correct in [("cuda", True, False), ("cuda", False, True), ("cpu", True, True)]:
            with self.subTest(device=device, finite=finite, correct=correct):
                scalar = lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: value))

                class Tensor:
                    def __init__(self):
                        self.device = SimpleNamespace(type=device)
                    def __matmul__(self, other):
                        return self
                    def __eq__(self, value):
                        return scalar(correct)

                torch = SimpleNamespace(float32="fp32", float16="fp16",
                                        inference_mode=contextlib.nullcontext,
                                        ones=lambda *args, **kwargs: Tensor(),
                                        isfinite=lambda result: scalar(finite),
                                        cuda=SimpleNamespace(reset_peak_memory_stats=lambda index: None,
                                                             synchronize=lambda index: None,
                                                             max_memory_allocated=lambda index: 1234))
                report = cli.smoke_report({"status": "passed"}, torch, "float32")
                self.assertEqual(report["status"], "failed")

    def test_saved_report_is_json_and_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                cli.write_report({"status": "blocked"}, path)
            original = path.read_bytes()
            self.assertEqual(json.loads(original)["status"], "blocked")
            with self.assertRaises(FileExistsError):
                cli.write_report({"status": "passed"}, path)
            self.assertEqual(path.read_bytes(), original)

    def test_existing_output_is_rejected_before_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("original", encoding="utf-8")
            with patch.object(cli, "environment_report") as probe, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(["doctor", "--output", str(path)])
            self.assertEqual(error.exception.code, 2)
            probe.assert_not_called()
            self.assertEqual(path.read_text(), "original")

    def test_blocked_result_returns_nonzero(self):
        with patch.object(cli, "environment_report", return_value=({"status": "blocked"}, None)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["doctor"]), 1)


if __name__ == "__main__":
    unittest.main()
