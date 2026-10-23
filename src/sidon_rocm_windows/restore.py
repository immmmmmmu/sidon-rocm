"""CPU/GPU comparison runs: the official sidon_demo_beta app.py pipeline on CPU and/or GPU.

Mirrors app.py at the Space revision pinned in config/upstream.json. GPU runs
never fall back to CPU, and existing outputs are never replaced.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CHUNK_SECONDS = 96  # app.py value
TAIL_PAD_16K = 24000
EDGE_PAD_16K = 160
DECODER_TRIM = 960
OUTPUT_RATE = 48000
COSINE_THRESHOLD = 0.99
# miopen: PyTorch default. native: PyTorch's own conv kernels, much faster for
# the decoder's 1-D convs on gfx1151 / Windows ROCm 7.2.1 (see docs/findings.md).
CONV_BACKENDS = ("miopen", "native")


class ModelFileError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_files(directory, files):
    for name, meta in files.items():
        path = Path(directory) / name
        if not path.is_file():
            raise ModelFileError(f"missing model file: {name}")
        if path.stat().st_size != meta["size"] or sha256_file(path) != meta["sha256"]:
            raise ModelFileError(f"model file does not match the pinned hash: {name}")
    return files


def read_wav(path):
    """Read 16-bit PCM WAV as int16-scaled floats shaped (samples, channels)."""
    import torch
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV input is supported")
        rate, channels = handle.getframerate(), handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    data = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32)
    return rate, data.view(-1, channels)


def write_wav(path, waveform, rate):
    """Write mono 16-bit PCM; values are clipped to [-1, 1] instead of wrapping."""
    import torch
    pcm = (waveform.detach().cpu().clamp(-1, 1) * 32767).round().to(torch.int16)
    with Path(path).open("xb") as raw, wave.open(raw, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.numpy().tobytes())


@dataclass
class Prepared:
    wav_16k: object
    target_samples: int


def prepare(waveform, sample_rate):
    """Normalize, downmix, high-pass, resample and tail-pad as app.py does."""
    import torch
    import torchaudio
    peak = waveform.abs().max()
    if peak == 0:
        raise ValueError("input is silent")
    waveform = 0.9 * (waveform / peak)
    target_samples = int(OUTPUT_RATE / sample_rate * waveform.shape[0])
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=1)
    wav = torchaudio.functional.highpass_biquad(waveform.view(1, -1), sample_rate, 50)
    wav_16k = torchaudio.functional.resample(wav, sample_rate, 16000)
    return Prepared(torch.nn.functional.pad(wav_16k, (0, TAIL_PAD_16K)), target_samples)


def _sync(device):
    import torch
    if device != "cpu":
        torch.cuda.synchronize(device)


def run_models(prepared, preprocessor, feature_extractor, decoder, device,
               chunk_seconds=DEFAULT_CHUNK_SECONDS):
    """Return the restored 48 kHz waveform on CPU and per-stage timings/devices."""
    import torch
    stats = {"preprocess_s": 0.0, "feature_extractor_s": 0.0, "decoder_s": 0.0,
             "feature_devices": set(), "decoder_output_devices": set(), "chunks": 0}
    restored, cache = [], None
    with torch.inference_mode():
        for chunk in prepared.wav_16k.view(-1).split(16000 * chunk_seconds):
            start = time.perf_counter()
            inputs = preprocessor(torch.nn.functional.pad(chunk, (EDGE_PAD_16K, EDGE_PAD_16K)),
                                  sampling_rate=16000, return_tensors="pt")
            features_in = inputs["input_features"].to(device)
            _sync(device)
            stats["preprocess_s"] += time.perf_counter() - start
            start = time.perf_counter()
            feature = feature_extractor(features_in)["last_hidden_state"]
            _sync(device)
            stats["feature_extractor_s"] += time.perf_counter() - start
            stats["feature_devices"].add(str(feature.device))
            if cache is not None:
                feature = torch.cat([cache, feature], dim=1)
            start = time.perf_counter()
            out = decoder(feature.transpose(1, 2))
            _sync(device)
            stats["decoder_s"] += time.perf_counter() - start
            stats["decoder_output_devices"].add(str(out.device))
            restored.append(out.reshape(-1)[:-DECODER_TRIM])
            cache = feature[:, -1:]
            stats["chunks"] += 1
    waveform = torch.cat(restored).float().cpu()[:prepared.target_samples]
    stats["feature_devices"] = sorted(stats["feature_devices"])
    stats["decoder_output_devices"] = sorted(stats["decoder_output_devices"])
    return waveform, stats


def output_checks(waveform):
    import torch
    finite = bool(torch.isfinite(waveform).all().item())
    clean = torch.nan_to_num(waveform)
    return {
        "samples": int(waveform.numel()),
        "finite": finite,
        "out_of_range_samples": int((clean.abs() > 1).sum().item()),
        "peak": float(clean.abs().max().item()) if waveform.numel() else 0.0,
        "rms": float(clean.pow(2).mean().sqrt().item()) if waveform.numel() else 0.0,
    }


def compare(reference, candidate):
    import torch
    n = min(reference.numel(), candidate.numel())
    ref, cand = reference[:n].double(), candidate[:n].double()
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item()
    noise = (ref - cand).pow(2).sum().item()
    snr = float("inf") if noise == 0 else 10 * torch.log10(ref.pow(2).sum() / noise).item()
    return {"samples_compared": n, "cosine_similarity": cosine,
            "snr_db": snr if snr != float("inf") else 999.0,
            "max_abs_diff": (ref - cand).abs().max().item()}


def error_summary(exc):
    """TorchScript errors start with a fixed banner; the root cause is the last line."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return {"type": type(exc).__name__,
            "message": lines[0][:300] if lines else "",
            "cause": lines[-1][:300] if lines else "",
            "detail": str(exc)[-2000:]}


def set_conv_backend(torch, backend):
    torch.backends.cudnn.enabled = backend == "miopen"


def run_device(device, root, upstream, prepared, chunk_seconds=DEFAULT_CHUNK_SECONDS):
    import torch
    from transformers import SeamlessM4TFeatureExtractor
    result = {"device": device, "status": "failed"}
    try:
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but torch.cuda is not available")
        suffix = "cpu" if device == "cpu" else "cuda"
        model_dir = root / "models" / "sidon-v0.1" / upstream["model"]["revision"]
        pre_dir = root / "models" / "w2v-bert-2.0" / upstream["preprocessor"]["revision"]
        preprocessor = SeamlessM4TFeatureExtractor.from_pretrained(str(pre_dir), local_files_only=True)
        if device != "cpu":
            # On the Windows ROCm wheel, resetting stats before the device is
            # initialized raises "Invalid device argument".
            torch.cuda.get_device_properties(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        fe = torch.jit.load(str(model_dir / f"feature_extractor_{suffix}.pt"), map_location=device).to(device)
        decoder = torch.jit.load(str(model_dir / f"decoder_{suffix}.pt"), map_location=device).to(device)
        _sync(device)
        result["model_load_s"] = time.perf_counter() - start
        start = time.perf_counter()
        waveform, stats = run_models(prepared, preprocessor, fe, decoder, device, chunk_seconds)
        result["inference_s"] = time.perf_counter() - start
        result.update(stats)
        if device != "cpu":
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            on_gpu = all(d.startswith("cuda") for d in stats["feature_devices"] + stats["decoder_output_devices"])
            if not on_gpu:
                raise RuntimeError("GPU run produced tensors outside the GPU")
        result["checks"] = output_checks(waveform)
        result["status"] = "passed" if result["checks"]["finite"] else "failed"
        return result, waveform
    except Exception as exc:
        result["error"] = error_summary(exc)
        return result, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="16-bit PCM WAV; never modified")
    parser.add_argument("--run-id", required=True, help="New name for data/output/<id> and reports/<id>")
    parser.add_argument("--devices", default="cpu,cuda:0", help="Comma-separated; cpu is the reference")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
                        help="16 kHz chunk length; app.py uses 96. Shorter chunks keep each GPU call short")
    parser.add_argument("--conv-backend", choices=CONV_BACKENDS, default="miopen",
                        help="GPU convolution kernels; native skips MIOpen")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be positive")
    out_dir, report_dir = args.root / "data" / "output" / args.run_id, args.root / "reports" / args.run_id
    if out_dir.exists() or report_dir.exists():
        print("run-id already used; choose a new one", file=sys.stderr)
        return 2
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    upstream = json.loads((args.root / "config" / "upstream.json").read_text(encoding="utf-8"))
    report = {"schema_version": 1, "stage": "compare",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "status": "failed",
              "model": {k: upstream["model"][k] for k in ("model_id", "revision")},
              "preprocessor_revision": upstream["preprocessor"]["revision"],
              "inference_reference_revision": upstream["inference_reference"]["revision"],
              "cosine_threshold": COSINE_THRESHOLD, "chunk_seconds": args.chunk_seconds, "conv_backend": args.conv_backend, "runs": [], "comparisons": {}}
    try:
        model_dir = args.root / "models" / "sidon-v0.1" / upstream["model"]["revision"]
        verify_model_files(model_dir, upstream["model"]["files"])
        pre = upstream["preprocessor"]
        verify_model_files(args.root / "models" / "w2v-bert-2.0" / pre["revision"], pre["files"])
        import torch
        import torchaudio
        import transformers
        set_conv_backend(torch, args.conv_backend)
        report["versions"] = {"torch": torch.__version__, "hip": torch.version.hip,
                              "torchaudio": torchaudio.__version__, "transformers": transformers.__version__}
        rate, waveform = read_wav(args.input)
        report["input"] = {"sha256": sha256_file(args.input), "sample_rate": rate,
                           "channels": waveform.shape[1], "seconds": waveform.shape[0] / rate}
        prepared = prepare(waveform, rate)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
        devices = []
    outputs = {}
    out_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    for device in devices:
        result, restored = run_device(device, args.root, upstream, prepared, args.chunk_seconds)
        if restored is not None:
            name = "restored-" + device.replace(":", "") + ".wav"
            write_wav(out_dir / name, restored, OUTPUT_RATE)
            result["output_file"] = name
            outputs[device] = restored
        report["runs"].append(result)
    if "cpu" in outputs:
        for device, restored in outputs.items():
            if device != "cpu":
                report["comparisons"][f"{device}_vs_cpu"] = compare(outputs["cpu"], restored)
    runs_ok = bool(devices) and all(r["status"] == "passed" for r in report["runs"])
    match_ok = all(c["cosine_similarity"] >= COSINE_THRESHOLD for c in report["comparisons"].values())
    report["status"] = "passed" if runs_ok and match_ok else "failed"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with (report_dir / "compare.json").open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(payload, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
