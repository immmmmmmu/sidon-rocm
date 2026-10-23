"""Download the pinned sidon-v0.1 weights and w2v-BERT 2.0 preprocessor config.

Files go to models/<name>/<revision>/ and are checked against the sha256 and
size in config/upstream.json. Verified files are not downloaded again, and an
existing file that does not match is reported, never overwritten.
"""

import argparse
import json
import sys
from pathlib import Path

from .restore import ModelFileError, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRS = {"model": "sidon-v0.1", "preprocessor": "w2v-bert-2.0"}


def load_upstream(root):
    return json.loads((root / "config" / "upstream.json").read_text(encoding="utf-8"))


def hf_downloader():
    from huggingface_hub import hf_hub_download

    def download(repo_id, filename, revision, local_dir):
        return hf_hub_download(repo_id, filename, revision=revision, local_dir=local_dir)
    return download


def matches(path, meta):
    return path.stat().st_size == meta["size"] and sha256_file(path) == meta["sha256"]


def fetch_all(root, upstream, download):
    results = []
    for key, folder in MODEL_DIRS.items():
        entry = upstream[key]
        directory = root / "models" / folder / entry["revision"]
        for name, meta in entry["files"].items():
            path = directory / name
            if path.exists():
                if not matches(path, meta):
                    raise ModelFileError(f"{path} exists but does not match the pinned hash; "
                                         "move it away and run fetch-models again")
                results.append({"file": name, "status": "present"})
                continue
            download(entry["model_id"], name, entry["revision"], str(directory))
            if not path.is_file() or not matches(path, meta):
                raise ModelFileError(f"downloaded {name} does not match the pinned hash")
            results.append({"file": name, "status": "downloaded"})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sidon-rocm fetch-models", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Checkout with config/ and models/")
    args = parser.parse_args(argv)
    try:
        for result in fetch_all(args.root, load_upstream(args.root), hf_downloader()):
            print(f"{result['status']:>10}  {result['file']}")
    except Exception as exc:
        print(f"fetch-models failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("all files match config/upstream.json")
    return 0
