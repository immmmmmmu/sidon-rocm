"""Restore one recording with the pinned sidon-v0.1 TorchScript.

Writes a new 48 kHz mono 16-bit WAV and a JSON report next to it. The input
is never modified, existing files are never replaced, and a GPU failure never
falls back to CPU (use --device cpu explicitly).
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .restore import (CONV_BACKENDS, DEFAULT_CHUNK_SECONDS, OUTPUT_RATE, error_summary, prepare,
                      read_wav, run_device, set_conv_backend, sha256_file, verify_model_files, write_wav)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def device_name(value):
    if value == "cpu" or (value.startswith("cuda:") and value[5:].isdigit()):
        return value
    raise argparse.ArgumentTypeError("use cpu or cuda:<index>")


def build_parser():
    parser = argparse.ArgumentParser(prog="sidon-rocm restore", description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="16-bit PCM WAV; never modified")
    parser.add_argument("--output", type=Path, required=True, help="New WAV path; existing files are refused")
    parser.add_argument("--report", type=Path, help="New JSON path (default: output with .json)")
    parser.add_argument("--device", type=device_name, default="cuda:0", help="cuda:0 (default) or cpu")
    parser.add_argument("--conv-backend", choices=CONV_BACKENDS, default="native",
                        help="native (default) avoids slow MIOpen convolutions and GPU timeouts on the EVO-X2")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Checkout with config/ and models/")
    return parser


def load_upstream(root):
    return json.loads((root / "config" / "upstream.json").read_text(encoding="utf-8"))


def verify_all(root, upstream):
    model, pre = upstream["model"], upstream["preprocessor"]
    verify_model_files(root / "models" / "sidon-v0.1" / model["revision"], model["files"])
    verify_model_files(root / "models" / "w2v-bert-2.0" / pre["revision"], pre["files"])


def same_file(a, b):
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds must be positive")
    report_path = args.report or args.output.with_suffix(".json")
    if not args.input.is_file():
        print("input not found", file=sys.stderr)
        return 2
    if same_file(args.input, args.output) or same_file(args.input, report_path):
        print("output must differ from the input", file=sys.stderr)
        return 2
    if args.output.exists() or report_path.exists():
        print("output or report already exists; choose a new name", file=sys.stderr)
        return 2

    report = {"schema_version": 1, "stage": "restore", "status": "failed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "device": args.device, "conv_backend": args.conv_backend, "chunk_seconds": args.chunk_seconds,
              "output": args.output.name}
    start = time.perf_counter()
    try:
        upstream = load_upstream(args.root)
        report["model"] = {k: upstream["model"].get(k) for k in ("model_id", "revision")}
        report["model"]["files"] = upstream["model"]["files"]
        report["preprocessor_revision"] = upstream["preprocessor"]["revision"]
        verify_all(args.root, upstream)
        import torch
        import torchaudio
        import transformers
        report["versions"] = {"torch": torch.__version__, "hip": torch.version.hip,
                              "torchaudio": torchaudio.__version__, "transformers": transformers.__version__}
        set_conv_backend(torch, args.conv_backend)
        rate, waveform = read_wav(args.input)
        report["input"] = {"sha256": sha256_file(args.input), "sample_rate": rate,
                           "channels": waveform.shape[1], "seconds": waveform.shape[0] / rate}
        prepared = prepare(waveform, rate)
        result, restored = run_device(args.device, args.root, upstream, prepared, args.chunk_seconds)
        report["run"] = result
        if result["status"] == "passed" and restored is not None:
            write_wav(args.output, restored, OUTPUT_RATE)
            report["status"] = "passed"
        elif "error" in result:
            report["error"] = result["error"]
        else:
            report["error"] = {"type": "OutputCheckError", "message": "restored audio failed output checks"}
    except Exception as exc:
        report["error"] = error_summary(exc)
    report["elapsed_s"] = time.perf_counter() - start
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["status"] == "passed":
        print(f"restored: {args.output} ({report['elapsed_s']:.1f}s, report {report_path})")
        return 0
    print(f"restore failed: {report['error'].get('type')}: {report['error'].get('message', '')} "
          f"(report {report_path})", file=sys.stderr)
    return 1
