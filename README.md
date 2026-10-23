# sidon-rocm-windows

Run the [Sidon](https://github.com/sarulab-speech/Sidon) speech restoration model on an AMD GPU under **native Windows with ROCm** — no WSL, no CUDA.

> **Unofficial.** This project is not affiliated with the Sidon authors or AMD. As of 2026-10, neither Sidon nor AMD documents a Windows AMD GPU setup for Sidon. This repository pins the published single-speaker model and adds the workarounds needed to make it run correctly and fast on Windows ROCm.

日本語の説明は [README.ja.md](README.ja.md) にあります。

## What it does

- Restores a speech recording with the published TorchScript model [`sarulab-speech/sidon-v0.1`](https://huggingface.co/sarulab-speech/sidon-v0.1), following the official demo's [`app.py`](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta/blob/cc6ac8f4b41c16a81d2a6984f8de1c4919a6c150/app.py) pipeline (16 kHz input features, 48 kHz mono output, 96 s chunks).
- Pins every model file by revision and sha256 and checks them before each run.
- Runs the decoder convolutions without MIOpen by default, which on the tested machine is about 7× faster and avoids Windows GPU timeouts (see [docs/findings.md](docs/findings.md)).
- Never overwrites the input or an existing output, never silently falls back to CPU, and refuses to write output that contains NaN/Inf.

## Tested environment

Only this combination has been tested. Other AMD GPUs supported by AMD's Windows PyTorch wheels may work but are unverified.

| Item | Version |
| --- | --- |
| Machine | GMKtec EVO-X2, Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB (64 GB assigned to the GPU in BIOS) |
| OS | Windows 11 Pro 25H2 (build 26200) |
| GPU driver | 32.0.32015.7029 |
| Python | 3.12.10 |
| PyTorch | 2.9.1+rocm7.2.1 (AMD Windows wheels), torchaudio 2.9.1, ROCm 7.2.1 |
| Preprocessing | transformers 4.53.3, huggingface-hub 0.34.0, tokenizers 0.21.2 (same as the official demo's lock file) |

## Setup (PowerShell)

1. Create an isolated venv (Python 3.12 is required by AMD's wheels):

   ```powershell
   git clone <this repository>
   cd sidon-rocm-windows
   .\scripts\bootstrap.ps1
   $Py = '.\.venv\Scripts\python.exe'
   ```

2. Install AMD's Windows ROCm PyTorch wheels ([AMD instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html)):

   ```powershell
   & $Py -m pip install --find-links https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ -c config/constraints-rocm.txt 'torch==2.9.1+rocm7.2.1' 'torchaudio==2.9.1+rocm7.2.1' 'torchvision==0.24.1+rocm7.2.1' 'rocm==7.2.1'
   ```

3. Install the preprocessing dependencies. Check the dry run first: it must not replace `torch`, `torchaudio` or `rocm`.

   ```powershell
   & $Py -m pip install --dry-run -c config/constraints-rocm.txt transformers==4.53.3 huggingface_hub==0.34.0 tokenizers==0.21.2
   & $Py -m pip install -c config/constraints-rocm.txt transformers==4.53.3 huggingface_hub==0.34.0 tokenizers==0.21.2
   & $Py -m pip check
   ```

4. Install this package so the `sidon-rocm` command is available (or run `scripts\sidon_rocm.py` instead):

   ```powershell
   & $Py -m pip install --no-deps -e .
   ```

5. Check the GPU, then download the model files (about 2 GB, verified by sha256):

   ```powershell
   .\.venv\Scripts\sidon-rocm.exe doctor
   .\.venv\Scripts\sidon-rocm.exe gpu-smoke
   .\.venv\Scripts\sidon-rocm.exe fetch-models
   ```

## Usage

Input must be 16-bit PCM WAV. Convert other formats first, for example `ffmpeg -i in.m4a -c:a pcm_s16le in.wav`.

```powershell
.\.venv\Scripts\sidon-rocm.exe restore --input data\input\in.wav --output data\output\out.wav
```

This writes `out.wav` (48 kHz mono 16-bit) and `out.json` (model revision and hashes, library versions, device, timings and output checks).

| Option | Default | Notes |
| --- | --- | --- |
| `--device` | `cuda:0` | ROCm uses PyTorch's `cuda` device names. Use `cpu` to run on the CPU; a GPU failure never falls back to CPU. |
| `--conv-backend` | `native` | `native` disables MIOpen for convolutions. `miopen` is PyTorch's default and is slow on the tested machine. |
| `--chunk-seconds` | `96` | Same as the official demo. With `--conv-backend miopen`, use `10` to avoid GPU timeouts. |
| `--report` | output with `.json` | Must not exist yet. |

Exit codes: `0` success; `1` model, input, computation or output-check failure (no output is written); `2` invalid arguments, output equal to the input, or an existing output/report.

Close other programs that use the GPU heavily (for example a local LLM server) while restoring.

`scripts\compare_cpu_gpu.py` restores the same file on CPU and GPU and reports their cosine similarity, which is how the results below were checked.

## Results on the tested machine

Japanese speech, `--conv-backend native`, 96 s chunks. Times are inference time of the first run in a fresh process: they include first-use kernel preparation but exclude model loading and hash checks. End to end, `sidon-rocm restore` took 95.6 s for the 10-minute file.

| Input | GPU | CPU (reference) | GPU vs CPU cosine similarity |
| --- | --- | --- | --- |
| 60 s | 8.4 s | 69.3 s | 0.9999999998 |
| 10 min | 87.5 s (RTF 0.146) | 741.0 s | 0.9999999997 |

Repeating the 10-minute run in the same process took 57.2 s and then 33.9 s (RTF 0.056). Peak GPU allocation was about 10.1 GB for 10 minutes of audio. Restored 10–60 s clips were checked by listening; this is a subjective check, not a formal quality evaluation.

## Known issues

- **MIOpen convolutions are slow and can hit the Windows GPU timeout.** With PyTorch's default MIOpen path, 96 s chunks of real speech triggered GPU driver resets (`LiveKernelEvent 141`) and once returned NaN output without raising an error. This is why `--conv-backend native` is the default.
- **First run in a process is slow** because kernels are prepared on first use.
- The model's TorchScript files have `cuda:0` baked in (upstream issues [#5](https://github.com/sarulab-speech/Sidon/issues/5), [#6](https://github.com/sarulab-speech/Sidon/issues/6)); with a single GPU, `cuda:0` is the AMD GPU under ROCm.
- Out-of-memory and missing-GPU failures are handled as non-zero exits in code but have not been reproduced on real hardware.

Details and measurements: [docs/findings.md](docs/findings.md).

## Licenses and credits

- This repository: MIT ([LICENSE](LICENSE)).
- Sidon by Wataru Nakata and Yuki Saito (SaruLab): code under MIT ([repository](https://github.com/sarulab-speech/Sidon)); the `sidon-v0.1` model is published under MIT ([model card](https://huggingface.co/sarulab-speech/sidon-v0.1)). Paper: [arXiv:2509.17052](https://arxiv.org/abs/2509.17052).
- Preprocessing configuration from [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0) (MIT).
- PyTorch and ROCm wheels are provided by AMD under their own licenses and are not redistributed here. Model files are downloaded from Hugging Face by `fetch-models`, not included in this repository.

Check the model card and the training data licenses yourself before using the output commercially.
