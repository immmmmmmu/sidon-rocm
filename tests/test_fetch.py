"""fetch-models checks with a fake downloader; no network access."""

import hashlib
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sidon_rocm_windows import cli, fetch

PAYLOAD = {"a.pt": b"weights-a", "b.pt": b"weights-b", "preprocessor_config.json": b"{}"}


def meta(name):
    data = PAYLOAD[name]
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


UPSTREAM = {
    "model": {"model_id": "org/model", "revision": "rev1", "files": {"a.pt": meta("a.pt"), "b.pt": meta("b.pt")}},
    "preprocessor": {"model_id": "org/pre", "revision": "rev2",
                     "files": {"preprocessor_config.json": meta("preprocessor_config.json")}},
}


class FakeDownloader:
    def __init__(self, corrupt=()):
        self.calls = []
        self.corrupt = set(corrupt)

    def __call__(self, repo_id, filename, revision, local_dir):
        self.calls.append((repo_id, filename, revision))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"corrupt" if filename in self.corrupt else PAYLOAD[filename])
        return str(path)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_downloads_pinned_revisions_into_revision_folders(self):
        downloader = FakeDownloader()
        results = fetch.fetch_all(self.root, UPSTREAM, downloader)
        self.assertIn(("org/model", "a.pt", "rev1"), downloader.calls)
        self.assertIn(("org/pre", "preprocessor_config.json", "rev2"), downloader.calls)
        self.assertTrue((self.root / "models" / "sidon-v0.1" / "rev1" / "a.pt").is_file())
        self.assertTrue((self.root / "models" / "w2v-bert-2.0" / "rev2" / "preprocessor_config.json").is_file())
        self.assertEqual({r["status"] for r in results}, {"downloaded"})

    def test_verified_files_are_not_downloaded_again(self):
        fetch.fetch_all(self.root, UPSTREAM, FakeDownloader())
        downloader = FakeDownloader()
        results = fetch.fetch_all(self.root, UPSTREAM, downloader)
        self.assertEqual(downloader.calls, [])
        self.assertEqual({r["status"] for r in results}, {"present"})

    def test_hash_mismatch_after_download_fails(self):
        with self.assertRaises(fetch.ModelFileError):
            fetch.fetch_all(self.root, UPSTREAM, FakeDownloader(corrupt={"b.pt"}))

    def test_existing_file_with_wrong_hash_is_not_overwritten(self):
        target = self.root / "models" / "sidon-v0.1" / "rev1" / "a.pt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"user file")
        downloader = FakeDownloader()
        with self.assertRaises(fetch.ModelFileError):
            fetch.fetch_all(self.root, UPSTREAM, downloader)
        self.assertEqual(target.read_bytes(), b"user file")
        self.assertNotIn(("org/model", "a.pt", "rev1"), downloader.calls)

    def test_main_returns_nonzero_on_failure(self):
        with unittest.mock.patch.object(fetch, "load_upstream", return_value=UPSTREAM), \
             unittest.mock.patch.object(fetch, "hf_downloader", return_value=FakeDownloader(corrupt={"a.pt"})), \
             unittest.mock.patch("sys.stdout"), unittest.mock.patch("sys.stderr"):
            self.assertEqual(fetch.main(["--root", str(self.root)]), 1)

    def test_sidon_rocm_fetch_models_dispatches(self):
        with unittest.mock.patch.object(fetch, "main", return_value=0) as target:
            self.assertEqual(cli.main(["fetch-models", "--root", "x"]), 0)
        target.assert_called_once_with(["--root", "x"])


if __name__ == "__main__":
    unittest.main()
