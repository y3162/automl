"""Versioned 16 kHz waveform pairs and sample-accurate training crops."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import soundfile as sf
import torch
from torchaudio.functional import resample

from automl.dataset import DatasetVersion, Sample
from automl.paths import installed_corpus_link
from corpora.index import INDEX
from examples.xor.components.catalog import CatalogDataset, check_digest, store_path

from .random import mix_seed
from .voicebank_snapshot import render_pair, snapshot_rows, source_path

CORPUS = "data/corpora/voicebank_demand"
MANIFEST = "data/work/asr_guided_se/voicebank28.jsonl"


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_manifest(root):
    corpus = root / CORPUS
    with duckdb.connect(str(root / INDEX), read_only=True) as db:
        rows = db.execute("""
            SELECT official_subset, utterance_id, clean_relative_path, noisy_relative_path, transcript
            FROM corpora.voicebank_demand WHERE official_subset IN ('trainset_28spk', 'testset')
            ORDER BY official_subset, utterance_id
        """).fetchall()
    if not rows:
        raise ValueError("The VoiceBank index contains no 28-speaker training or test pairs")

    def record(row):
        subset, uid, clean, noisy, transcript = row
        return json.dumps(dict(id=uid, split="train" if subset == "trainset_28spk" else "test",
                               clean=clean, noisy=noisy, transcript=transcript,
                               clean_sha256=_digest(corpus / clean), noisy_sha256=_digest(corpus / noisy)),
                          ensure_ascii=False, sort_keys=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return _write_manifest(root / MANIFEST, pool.map(record, rows))


def _write_manifest(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        try:
            for line in lines:
                tmp.write(line + "\n")
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return path


def _manifest(root, path, digest):
    if digest:
        check_digest(digest)
        saved = store_path(root) / "inputs" / digest
        if saved.is_file():
            return saved
    return root / path


class PairVersion(DatasetVersion):
    def __init__(self, manifest, corpus):
        self.corpus = corpus
        self.rows = {}
        self._verified = {}
        with manifest.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                uid = row["id"]
                if uid in self.rows:
                    raise ValueError(f"Duplicate audio sample {uid}")
                self.rows[uid] = row

    def sample_ids(self, split):
        return [uid for uid, row in self.rows.items() if row["split"] == split]

    def _audio(self, row, field):
        path = self.corpus / row[field]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row[field + "_sha256"]:
            raise ValueError(f"Audio checksum mismatch for {path}")
        values, rate = sf.read(BytesIO(content), dtype="float32", always_2d=True)
        if values.shape[1] != 1 or not len(values):
            raise ValueError(f"Expected nonempty mono audio in {path}")
        return resample(torch.from_numpy(values[:, 0]), rate, 16000)

    def verify_asset(self, path, digest):
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, digest)
        if self._verified.get(path) != identity:
            if _digest(path) != digest:
                raise ValueError(f"Audio checksum mismatch for {path}")
            self._verified[path] = identity

    def get_sample(self, sample_id):
        row = self.rows[sample_id]
        if "sources" in row:
            clean, noisy = render_pair(self.corpus, row, self.verify_asset)
        else:
            clean, noisy = self._audio(row, "clean"), self._audio(row, "noisy")
        if clean.shape != noisy.shape:
            raise ValueError(f"Waveform lengths differ for {sample_id}")
        scale = noisy.abs().max() + 1e-9
        return Sample(sample_id, (clean / scale, noisy / scale), {"transcript": row["transcript"]})

    def iter_samples(self, *, split):
        for uid in self.sample_ids(split):
            yield self.get_sample(uid)


class VoiceBankPairs(CatalogDataset):
    @classmethod
    def resolve_settings(cls, settings, root):
        if set(settings) - {"manifest", "corpus", "sha256"}:
            raise ValueError("Unknown audio dataset settings")
        manifest = settings.get("manifest", MANIFEST)
        corpus = settings.get("corpus", CORPUS)
        expected = settings.get("sha256")
        path = _manifest(root, manifest, expected)
        if not path.is_file():
            raise ValueError(f"Missing {path}. Prepare the VoiceBank pair manifest with python -m methods.asr_guided_se.v1.data")
        digest = _digest(path)
        if expected is not None and expected != digest:
            raise ValueError("Audio manifest checksum mismatch")
        return dict(manifest=manifest, corpus=corpus, sha256=digest)

    @classmethod
    def replay_files(cls, settings, root):
        digest = settings["sha256"]
        return {digest: _manifest(root, settings["manifest"], digest)}

    @classmethod
    def build(cls, settings, *, root):
        return cls(root, **cls.resolve_settings(settings, root))

    def __init__(self, root, manifest, corpus, sha256):
        self.root = root
        self.manifest = manifest
        self.corpus = (root / corpus).resolve()
        super().__init__(store_path(root) / "datasets/asr_guided_se/catalog.sqlite3", sha256)
        # Direct Dataset API callers also retain the immutable manifest.
        dest = store_path(root) / "inputs" / sha256
        if not dest.exists():
            content = (root / manifest).read_bytes()
            if hashlib.sha256(content).hexdigest() != sha256:
                raise ValueError("Audio manifest changed during registration")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                tmp.write(content)
            os.replace(tmp.name, dest)

    def open_version(self, version_id):
        check_digest(version_id)
        path = _manifest(self.root, self.manifest, version_id)
        if _digest(path) != version_id:
            raise ValueError("Audio manifest version mismatch")
        return PairVersion(path, self.corpus)

    @classmethod
    def verify_version(cls, settings, root, version_id):
        checked = cls.resolve_settings({**settings, "sha256": version_id}, root)
        version = PairVersion(_manifest(root, checked["manifest"], version_id), root / checked["corpus"])
        for row in version.rows.values():
            if "sources" in row:
                for source in row["sources"].values():
                    version.verify_asset(source_path(version.corpus, source), source["sha256"])
            else:
                for field in ("clean", "noisy"):
                    version.verify_asset(version.corpus / row[field], row[field + "_sha256"])


@dataclass
class WaveBatch:
    clean: torch.Tensor
    noisy: torch.Tensor
    references: list[torch.Tensor]
    spans: list[tuple[int, int, int, int]]

    @property
    def lengths(self):
        return torch.tensor([s[1] for s in self.spans], device=self.clean.device, dtype=torch.long)


def crop_batch(samples, window, seed, epoch, device):
    clean_rows, noisy_rows, references, spans = [], [], [], []
    for sample in samples:
        clean, noisy = sample.data
        full = len(clean)
        start = mix_seed("crop", seed, epoch, sample.sample_id) % (max(full - window, 0) + 1)
        content = min(full, window)
        pad = (0, window - content)
        clean_rows.append(torch.nn.functional.pad(clean[start:start + content], pad))
        noisy_rows.append(torch.nn.functional.pad(noisy[start:start + content], pad))
        references.append(clean)
        spans.append((start, content, window, full))
    return WaveBatch(torch.stack(clean_rows).to(device), torch.stack(noisy_rows).to(device), references, spans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze the indexed VoiceBank 28-speaker WAV pairs and transcripts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path, help="audio-db VoiceBank SNR-shift snapshot directory")
    parser.add_argument("--corpora", type=Path, help="parent of VCTK, Babble, SSN and DEMAND48")
    args = parser.parse_args()
    if args.snapshot is None and args.corpora is None:
        print(prepare_manifest(args.root))
    elif args.snapshot is not None and args.corpora is not None:
        rows = list(snapshot_rows(args.snapshot, args.corpora))
        shift = rows[0]["snr_shift_db"]
        shift_name = f"{shift:g}".replace("-", "m")
        path = args.root / "data/work/asr_guided_se" / f"voicebank_{shift_name}db.jsonl"
        link = args.root / "data/corpora/asr_guided_se_sources"
        link.parent.mkdir(parents=True, exist_ok=True)
        with installed_corpus_link(link, args.corpora.resolve(strict=True)):
            print(_write_manifest(path, (json.dumps(row, sort_keys=True) for row in rows)))
    else:
        parser.error("--snapshot and --corpora must be provided together")
