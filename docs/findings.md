# Findings: Sidon on Windows ROCm (gfx1151)

Measurements from one machine (GMKtec EVO-X2, Ryzen AI Max+ 395 / Radeon 8060S gfx1151, Windows 11 25H2, driver 32.0.32015.7029, PyTorch 2.9.1+rocm7.2.1), 2026-10-22 to 2026-10-23. Inputs were Japanese single-speaker recordings cut from one longer recording. GPU results were always compared with the same pipeline run on CPU using the `*_cpu.pt` TorchScript files.

## 1. Pipeline

The published `sidon-v0.1` repository (revision `b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52`) contains four TorchScript files: `feature_extractor_{cpu,cuda}.pt` and `decoder_{cpu,cuda}.pt`. The official demo [`app.py`](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta/blob/cc6ac8f4b41c16a81d2a6984f8de1c4919a6c150/app.py) defines the inference steps, which this project mirrors in `src/sidon_rocm_windows/restore.py`:

1. Peak-normalize to 0.9, average channels to mono.
2. 50 Hz high-pass at the input sample rate, resample to 16 kHz, append 1.5 s of silence.
3. Split into 96 s chunks, pad each chunk by 160 samples on both sides.
4. Compute w2v-BERT 2.0 mel features on CPU (`SeamlessM4TFeatureExtractor`).
5. Feature extractor → prepend the last feature frame of the previous chunk → decoder.
6. Drop the last 960 samples of each chunk, concatenate, trim to the input duration at 48 kHz.

The preprocessing libraries are pinned to the demo's `uv.lock` (transformers 4.53.3) so that features match the demo.

## 2. GPU timeouts with MIOpen (TDR)

With PyTorch's default convolution path (MIOpen) and the demo's 96 s chunks:

| Input | GPU result |
| --- | --- |
| 10 s | OK |
| 30 s | failed once inside the TorchScript interpreter, succeeded on retry |
| 60 s | failed three times; once it finished without an exception but returned NaN/Inf for 764,129 samples |

Every failing run coincided with `LiveKernelEvent 141` in Windows Error Reporting (the GPU stopped responding and Windows reset the driver). Successful runs had none. Stopping another GPU-heavy program did not change the 60 s result, so the cause is the length of a single GPU call, not contention.

Consequences:

- Always check outputs for NaN/Inf. A driver reset can produce garbage without an exception. `restore` refuses to write such output.
- With MIOpen, 10 s chunks avoided the timeouts (60 s input, cosine similarity 0.9999999998 against CPU).

## 3. MIOpen is the bottleneck; native convolutions fix speed and timeouts

Profiling a warm 10 s chunk showed 96% of decoder GPU time in `aten::miopen_convolution` (32 calls, about 100 ms each). MIOpen also logs `GemmFwdRest` workspace warnings ("workspace required: 741853056, provided ptr: 0 size: 0").

| Setting (60 s input, 10 s chunks, same process, 3rd run) | Time |
| --- | --- |
| MIOpen (default) | 19.8 s |
| MIOpen with `torch.backends.cudnn.benchmark = True` | 19.9 s |
| `torch.backends.cudnn.enabled = False` (PyTorch native convolutions) | 2.8 s |

With native convolutions (`--conv-backend native`), the demo's 96 s chunks no longer caused driver resets:

| Run (fresh process) | Chunks | Convolutions | GPU inference | vs CPU | Driver resets |
| --- | --- | --- | --- | --- | --- |
| 60 s | 96 s | MIOpen | NaN output | — | yes |
| 60 s | 10 s | MIOpen | 330.7 s | 0.9999999998 | no |
| 60 s | 10 s | native | 52.5 s | 0.9999999998 | no |
| 60 s | 96 s | native | 8.4 s | 0.9999999998 | no |
| 10 min | 96 s | native | 87.5 s | 0.9999999997 | no |

The 10-minute run peaked at about 10.1 GB of GPU memory. Repeating it in one process took 87.0 s, 57.2 s, then 33.9 s, with identical output. At the six 96 s chunk joins, the sample-to-sample difference was at most 2% of the file's 99.99th percentile, so no discontinuities were detected (a numeric check, not a listening test).

## 4. Smaller Windows ROCm quirks

- `torch.cuda.reset_peak_memory_stats(...)` raises `RuntimeError: Invalid device argument` if it is the first GPU call in the process. Query the device first (for example `torch.cuda.get_device_properties(0)`).
- TorchScript errors start with a fixed banner ("The following operation failed in the TorchScript interpreter."); the actual cause is the last line of the message. `restore` records both.
- The TorchScript files have `cuda:0` constants baked in (upstream issues #5 and #6). With one GPU, `cuda:0` is the AMD GPU under ROCm, so this did not matter here.

## 5. Not verified

- Other AMD GPUs, other drivers, other ROCm versions (ROCm 10.1.0 lists Windows PyTorch 2.14.0, untested here).
- Out-of-memory and missing-GPU failures on real hardware.
- Formal quality evaluation (for example ASR error rates). Quality was only checked by listening to 10–60 s clips.
