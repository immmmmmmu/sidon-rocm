"""sidon-rocm: environment probes, model download and Sidon restoration on Windows ROCm."""

import argparse
import importlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def environment_report():
    report = {
        "schema_version": 1,
        "stage": "environment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.system(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "status": "blocked",
        "checks": {},
        "scope": "Environment only; Sidon inference has not been tested.",
    }
    checks = report["checks"]
    checks["windows_11"] = report["os"] == "Windows" and report["os_release"] == "11"
    checks["python_312"] = sys.version_info[:2] == (3, 12)
    try:
        torch = importlib.import_module("torch")
        report["torch"] = str(torch.__version__)
        report["hip"] = getattr(torch.version, "hip", None)
        checks["pinned_torch"] = report["torch"] == "2.9.1+rocm7.2.1"
        checks["hip_backend"] = bool(report["hip"])
        checks["gpu_available"] = bool(torch.cuda.is_available())
        if checks["gpu_available"]:
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {
                "index": 0,
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": props.total_memory,
                "architecture": getattr(props, "gcnArchName", None),
            }
            checks["radeon_8060s"] = "8060S" in report["gpu"]["name"].upper()
            # Some Windows wheels do not expose gcnArchName. Record that gap.
            arch = report["gpu"]["architecture"]
            checks["gfx1151"] = arch.split(":")[0] == "gfx1151" if arch else None
        else:
            checks["radeon_8060s"] = False
            checks["gfx1151"] = None
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "reason": "torch_import_or_device_query_failed"}
        return report, None
    required = ["windows_11", "python_312", "pinned_torch", "hip_backend", "gpu_available", "radeon_8060s"]
    if all(checks[k] is True for k in required) and checks["gfx1151"] is not False:
        report["status"] = "passed"
    return report, torch


def smoke_report(environment, torch, dtype):
    report = dict(environment)
    report.update(stage="gpu_smoke", status="blocked", dtype=dtype,
                  scope="Small matrix multiplication only; Sidon inference has not been tested.")
    if environment["status"] != "passed":
        report["reason"] = "environment_gate_failed"
        return report
    try:
        torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats(0)
            a = torch.ones((128, 128), device="cuda:0", dtype=torch_dtype)
            torch.cuda.synchronize(0)
            start = time.perf_counter()
            result = a @ a
            torch.cuda.synchronize(0)
            report["elapsed_ms"] = (time.perf_counter() - start) * 1000
            report["result_device"] = str(result.device)
            on_gpu = result.device.type == "cuda"
            finite = bool(torch.isfinite(result).all().item())
            # Every element has the independently known value 128.
            correct = bool((result == 128).all().item())
            report["computation_checks"] = {"on_gpu": on_gpu, "finite": finite, "correct": correct}
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["status"] = "passed" if on_gpu and finite and correct else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "reason": "gpu_computation_failed"}
    return report


def write_report(report, output):
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        # Exclusive creation: never replace an earlier run or a user file.
        with Path(output).open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["restore"]:
        from . import restore_cli
        return restore_cli.main(argv[1:])
    if argv[:1] == ["fetch-models"]:
        from . import fetch
        return fetch.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect the pinned Windows ROCm environment")
    doctor.add_argument("--output", type=Path, help="New JSON file; existing files are refused")
    smoke = sub.add_parser("gpu-smoke", help="Run a small matrix multiplication on the target GPU")
    smoke.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    smoke.add_argument("--output", type=Path, help="New JSON file; existing files are refused")
    sub.add_parser("fetch-models", help="Download and verify the pinned model files", add_help=False)
    sub.add_parser("restore", help="Restore one WAV with sidon-v0.1 (see: restore --help)", add_help=False)
    args = parser.parse_args(argv)
    # Refuse an existing destination before allocating any GPU memory.
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new report filename")
    report, torch = environment_report()
    if args.command == "gpu-smoke":
        report = smoke_report(report, torch, args.dtype)
    try:
        write_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"report_write_failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "passed" else 1
