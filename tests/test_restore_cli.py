"""restore CLI acceptance checks; fake models are not Sidon or GPU evidence."""

import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sidon_rocm_windows import cli, restore, restore_cli

HAS_TORCH = importlib.util.find_spec("torch") is not None
NEEDS_TORCH = unittest.skipUnless(HAS_TORCH, "torch is only installed in the project venv")


def run(argv):
    with unittest.mock.patch("sys.stderr"), unittest.mock.patch("sys.stdout"):
        return restore_cli.main(argv)


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "in.wav"
        self.source.write_bytes(b"original")

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_equal_to_input_is_refused(self):
        code = run(["--input", str(self.source), "--output", str(self.root / "." / "in.wav"), "--root", str(self.root)])
        self.assertEqual(code, 2)
        self.assertEqual(self.source.read_bytes(), b"original")

    def test_existing_output_or_report_is_refused(self):
        out = self.root / "out.wav"
        out.write_bytes(b"keep")
        self.assertEqual(run(["--input", str(self.source), "--output", str(out), "--root", str(self.root)]), 2)
        self.assertEqual(out.read_bytes(), b"keep")
        out.unlink()
        (self.root / "out.json").write_text("{}")
        self.assertEqual(run(["--input", str(self.source), "--output", str(out), "--root", str(self.root)]), 2)
        self.assertFalse(out.exists())

    def test_missing_input_is_refused(self):
        code = run(["--input", str(self.root / "nope.wav"), "--output", str(self.root / "o.wav"), "--root", str(self.root)])
        self.assertEqual(code, 2)

    def test_missing_models_fail_without_writing_output(self):
        out = self.root / "out.wav"
        with unittest.mock.patch.object(restore_cli, "load_upstream", return_value={
                "model": {"model_id": "m", "revision": "r", "files": {"a.pt": {"size": 1, "sha256": "0"}}},
                "preprocessor": {"revision": "p", "files": {}}}):
            code = run(["--input", str(self.source), "--output", str(out), "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        report = json.loads((self.root / "out.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "ModelFileError")
        self.assertEqual(self.source.read_bytes(), b"original")

    def test_device_defaults_to_gpu_with_native_convolutions(self):
        args = restore_cli.build_parser().parse_args(["--input", "a.wav", "--output", "b.wav"])
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.conv_backend, "native")
        self.assertEqual(args.chunk_seconds, 96)


class SubcommandTests(unittest.TestCase):
    def test_sidon_rocm_restore_dispatches_to_restore_cli(self):
        with unittest.mock.patch.object(restore_cli, "main", return_value=0) as target:
            code = cli.main(["restore", "--input", "a.wav", "--output", "b.wav"])
        self.assertEqual(code, 0)
        target.assert_called_once_with(["--input", "a.wav", "--output", "b.wav"])


@NEEDS_TORCH
class RestoreTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "in.wav"
        restore.write_wav(self.source, 0.5 * torch.sin(torch.arange(16000.0)), 16000)
        self.original = self.source.read_bytes()
        patcher = unittest.mock.patch.multiple(
            restore_cli, load_upstream=unittest.mock.DEFAULT, verify_all=unittest.mock.DEFAULT)
        mocks = patcher.start()
        mocks["load_upstream"].return_value = {"model": {"model_id": "m", "revision": "r", "files": {}},
                                               "preprocessor": {"revision": "p", "files": {}}}
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def fake_run(self, waveform, status="passed"):
        def run_device(device, root, upstream, prepared, chunk_seconds):
            result = {"device": device, "status": status, "checks": restore.output_checks(waveform)}
            return result, waveform
        return unittest.mock.patch.object(restore_cli, "run_device", side_effect=run_device)

    def test_success_writes_48k_output_and_report(self):
        out = self.root / "out.wav"
        with self.fake_run(self.torch.zeros(48000) + 0.1):
            code = run(["--input", str(self.source), "--output", str(out), "--root", str(self.root)])
        self.assertEqual(code, 0)
        rate, data = restore.read_wav(out)
        self.assertEqual((rate, data.shape), (48000, (48000, 1)))
        report = json.loads((self.root / "out.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["device"], "cuda:0")
        self.assertEqual(report["conv_backend"], "native")
        self.assertEqual(report["model"]["revision"], "r")
        self.assertIn("sha256", report["input"])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_non_finite_output_is_not_written(self):
        out = self.root / "out.wav"
        bad = self.torch.full((48000,), float("nan"))
        with self.fake_run(bad, status="failed"):
            code = run(["--input", str(self.source), "--output", str(out), "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())

    def test_gpu_failure_does_not_fall_back_to_cpu(self):
        out = self.root / "out.wav"
        calls = []
        def run_device(device, root, upstream, prepared, chunk_seconds):
            calls.append(device)
            return {"device": device, "status": "failed", "error": {"type": "RuntimeError"}}, None
        with unittest.mock.patch.object(restore_cli, "run_device", side_effect=run_device):
            code = run(["--input", str(self.source), "--output", str(out), "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["cuda:0"])
        self.assertFalse(out.exists())

    def test_silent_input_is_rejected(self):
        silent = self.root / "silent.wav"
        restore.write_wav(silent, self.torch.zeros(16000), 16000)
        out = self.root / "out.wav"
        code = run(["--input", str(silent), "--output", str(out), "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())
        report = json.loads((self.root / "out.json").read_text(encoding="utf-8"))
        self.assertIn("silent", report["error"]["message"])

    def test_non_pcm16_input_is_rejected(self):
        bad = self.root / "bad.wav"
        bad.write_bytes(b"not a wav")
        out = self.root / "out.wav"
        code = run(["--input", str(bad), "--output", str(out), "--root", str(self.root)])
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())


@NEEDS_TORCH
class PrepareEdgeTests(unittest.TestCase):
    def test_silence_raises_instead_of_dividing_by_zero(self):
        import torch
        with self.assertRaises(ValueError):
            restore.prepare(torch.zeros(16000, 1), 16000)

    def test_very_short_input_keeps_its_duration(self):
        import torch
        prepared = restore.prepare(torch.ones(800, 1) * 0.3, 16000)  # 50 ms
        self.assertEqual(prepared.target_samples, 2400)


if __name__ == "__main__":
    unittest.main()
