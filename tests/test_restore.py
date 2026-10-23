"""Regression checks for the restore pipeline; fake models are not Sidon or GPU evidence."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sidon_rocm_windows import restore

HAS_TORCH = importlib.util.find_spec("torch") is not None
NEEDS_TORCH = unittest.skipUnless(HAS_TORCH, "torch is only installed in the project venv")


def write_model_files(directory, contents):
    meta = {}
    for name, data in contents.items():
        (directory / name).write_bytes(data)
        meta[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return meta


class ModelFileTests(unittest.TestCase):
    def test_matching_files_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = write_model_files(Path(tmp), {"a.pt": b"abc", "b.pt": b"de"})
            self.assertEqual(restore.verify_model_files(Path(tmp), meta), meta)

    def test_mismatch_and_missing_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = write_model_files(Path(tmp), {"a.pt": b"abc"})
            (Path(tmp) / "a.pt").write_bytes(b"abd")
            with self.assertRaises(restore.ModelFileError):
                restore.verify_model_files(Path(tmp), meta)
            (Path(tmp) / "a.pt").unlink()
            with self.assertRaises(restore.ModelFileError):
                restore.verify_model_files(Path(tmp), meta)


class ErrorSummaryTests(unittest.TestCase):
    def test_torchscript_error_keeps_the_root_cause_at_the_end(self):
        exc = RuntimeError("The following operation failed in the TorchScript interpreter.\n"
                           "Traceback of TorchScript, serialized code:\n  File \"code/x.py\", line 9\n\n"
                           "RuntimeError: HIP error: out of memory\n")
        summary = restore.error_summary(exc)
        self.assertEqual(summary["type"], "RuntimeError")
        self.assertTrue(summary["message"].startswith("The following operation failed"))
        self.assertEqual(summary["cause"], "RuntimeError: HIP error: out of memory")
        self.assertLessEqual(len(summary["detail"]), 2000)


class CommandLineTests(unittest.TestCase):
    def test_existing_run_directory_is_refused_before_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reports" / "run-1").mkdir(parents=True)
            source = root / "in.wav"
            source.write_bytes(b"original")
            code = restore.main(["--input", str(source), "--run-id", "run-1", "--root", str(root)])
            self.assertEqual(code, 2)
            self.assertEqual(source.read_bytes(), b"original")

    def test_unknown_conv_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx, \
                 unittest.mock.patch("sys.stderr"):
                restore.main(["--input", "x.wav", "--run-id", "r", "--root", tmp, "--conv-backend", "cudnn"])
            self.assertEqual(ctx.exception.code, 2)

    def test_conv_backend_switches_miopen(self):
        torch = SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)))
        restore.set_conv_backend(torch, "native")
        self.assertFalse(torch.backends.cudnn.enabled)
        restore.set_conv_backend(torch, "miopen")
        self.assertTrue(torch.backends.cudnn.enabled)

    def test_non_positive_chunk_seconds_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as ctx, \
                 unittest.mock.patch("sys.stderr"):
                restore.main(["--input", "x.wav", "--run-id", "r", "--root", tmp, "--chunk-seconds", "0"])
            self.assertEqual(ctx.exception.code, 2)
            self.assertFalse((Path(tmp) / "reports" / "r").exists())


class FakeFeatureExtractor:
    """Returns one 1024-dim frame per two mel frames, like w2v-BERT 2.0."""

    def __init__(self, torch):
        self.torch = torch
        self.inputs = []

    def __call__(self, features):
        self.inputs.append(features.shape)
        frames = features.shape[1]
        return {"last_hidden_state": self.torch.zeros(1, frames, 1024, device=features.device)}


class FakeDecoder:
    """Emits 960 samples (48 kHz) per 50 Hz feature frame."""

    def __init__(self, torch):
        self.torch = torch
        self.frames = []

    def __call__(self, feature):
        self.frames.append(feature.shape[2])
        return self.torch.full((1, 1, feature.shape[2] * 960), 0.5, device=feature.device)


class FakePreprocessor:
    def __call__(self, audio, sampling_rate, return_tensors):
        import torch
        # 100 Hz mel frames stacked by stride 2: 160 samples per 80-dim pair.
        frames = audio.shape[-1] // 320
        return {"input_features": torch.zeros(1, frames, 160)}


@NEEDS_TORCH
class PipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_output_matches_input_duration_at_48k(self):
        wave = self.torch.sin(self.torch.arange(44100 * 3, dtype=self.torch.float32))
        prepared = restore.prepare(wave.view(-1, 1), 44100)
        self.assertEqual(prepared.target_samples, 48000 * 3)
        self.assertEqual(prepared.wav_16k.shape[-1], 16000 * 3 + 24000)
        self.assertAlmostEqual(prepared.wav_16k.abs().max().item(), 0.9, delta=0.2)
        out, _ = restore.run_models(prepared, FakePreprocessor(), FakeFeatureExtractor(self.torch),
                                    FakeDecoder(self.torch), "cpu")
        self.assertEqual(out.shape[-1], prepared.target_samples)
        self.assertEqual(out.device.type, "cpu")

    def test_stereo_is_averaged_to_mono(self):
        left = self.torch.ones(16000)
        prepared = restore.prepare(self.torch.stack([left, -left], dim=1), 16000)
        self.assertEqual(prepared.wav_16k.dim(), 2)
        self.assertEqual(prepared.wav_16k.shape[0], 1)

    def test_long_input_is_split_into_96_second_chunks_with_feature_cache(self):
        wave = self.torch.rand(16000 * 100) - 0.5
        prepared = restore.prepare(wave.view(-1, 1), 16000)
        fe, decoder = FakeFeatureExtractor(self.torch), FakeDecoder(self.torch)
        out, _ = restore.run_models(prepared, FakePreprocessor(), fe, decoder, "cpu")
        self.assertEqual(len(decoder.frames), 2)
        # The second chunk is prefixed with the last feature frame of the first.
        second_chunk_frames = fe.inputs[1][1]
        self.assertEqual(decoder.frames[1], second_chunk_frames + 1)
        self.assertEqual(out.shape[-1], prepared.target_samples)

    def test_chunk_seconds_controls_the_split(self):
        wave = self.torch.rand(16000 * 60) - 0.5
        prepared = restore.prepare(wave.view(-1, 1), 16000)
        decoder = FakeDecoder(self.torch)
        out, stats = restore.run_models(prepared, FakePreprocessor(), FakeFeatureExtractor(self.torch),
                                        decoder, "cpu", chunk_seconds=10)
        # 60 s + 1.5 s tail pad -> 7 chunks of at most 10 s.
        self.assertEqual(len(decoder.frames), 7)
        self.assertEqual(stats["chunks"], 7)
        self.assertEqual(out.shape[-1], prepared.target_samples)

    def test_output_checks_flag_non_finite_and_clipping(self):
        good = restore.output_checks(self.torch.tensor([0.1, -0.2, 0.3]))
        self.assertTrue(good["finite"])
        self.assertEqual(good["out_of_range_samples"], 0)
        bad = restore.output_checks(self.torch.tensor([float("nan"), 1.5, -2.0]))
        self.assertFalse(bad["finite"])
        self.assertEqual(bad["out_of_range_samples"], 2)

    def test_comparison_reports_cosine_and_snr(self):
        ref = self.torch.sin(self.torch.linspace(0, 100, 48000))
        same = restore.compare(ref, ref.clone())
        self.assertAlmostEqual(same["cosine_similarity"], 1.0, places=6)
        noisy = restore.compare(ref, ref + 0.5 * self.torch.randn(48000))
        self.assertLess(noisy["cosine_similarity"], 0.99)
        self.assertLess(noisy["snr_db"], same["snr_db"])

    def test_pcm16_wav_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.wav"
            wave = self.torch.tensor([0.0, 0.5, -0.5, 1.2])
            restore.write_wav(path, wave, 48000)
            rate, read = restore.read_wav(path)
            self.assertEqual(rate, 48000)
            self.assertEqual(read.shape, (4, 1))
            self.assertAlmostEqual(read[3, 0].item(), 32767, delta=1)  # clipped, not wrapped
            with self.assertRaises(FileExistsError):
                restore.write_wav(path, wave, 48000)


if __name__ == "__main__":
    unittest.main()
