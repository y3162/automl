"""Versioned MNIST partitions backed by the external IDX corpus."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from automl.dataset import DatasetVersion, Sample
from examples.xor.components.catalog import CatalogDataset, check_digest, connect, json_bytes, store_path

from .prepare import CORPUS, FILES, HEIGHT, IMAGE_HEADER, INDEX, WIDTH


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _index_path(root: Path, digest: str | None) -> Path:
    if digest is not None:
        check_digest(digest)
        retained = store_path(root) / "inputs" / digest
        if retained.is_file():
            return retained
    path = root / INDEX
    if not path.is_file():
        raise ValueError("MNIST metadata is missing. Configure .env and run make setup-data")
    return path


def _resources(index: Path) -> dict[str, tuple[int, str]]:
    with connect(index, read_only=True) as db:
        resources = {name: (count, digest) for name, count, digest in
                     db.execute("SELECT filename, sample_count, sha256 FROM files")}
    if set(resources) != {name for pair in FILES.values() for name in pair}:
        raise ValueError("Incomplete MNIST file index")
    return resources


def _check_corpus(corpus: Path, resources: dict) -> None:
    for filename, (_, expected) in resources.items():
        if _digest(corpus / filename) != expected:
            raise ValueError(f"MNIST checksum mismatch for {corpus / filename}")


def _version(resources: dict, validation_size: int) -> str:
    return hashlib.sha256(json_bytes({"files": resources, "validation_size": validation_size})).hexdigest()


def _catalog(root: Path) -> Path:
    return store_path(root) / "datasets/mnist/catalog.sqlite3"


class MnistVersion(DatasetVersion):
    def __init__(self, index: Path, corpus: Path, resources: dict, validation_size: int):
        self.index = index
        self.images = {}
        for partition, (filename, _) in FILES.items():
            count = resources[filename][0]
            self.images[partition] = np.memmap(corpus / filename, dtype=np.uint8, mode="r",
                                               offset=IMAGE_HEADER.size, shape=(count, 1, HEIGHT, WIDTH))
        train_count = len(self.images["train"])
        split_at = train_count - validation_size
        self.partitions = {
            "train": ("train", 0, split_at),
            "validation": ("train", split_at, train_count),
            "test": ("test", 0, len(self.images["test"])),
        }

    def _sample(self, partition: str, idx: int, label: int) -> Sample:
        pixels = self.images[partition][idx].astype(np.float32) / np.float32(255)
        return Sample(f"{partition}/{idx}", pixels, {"label": label})

    def iter_samples(self, *, split: str) -> Iterator[Sample]:
        partition, begin, end = self.partitions[split]
        with connect(self.index, read_only=True) as db:
            rows = db.execute("""SELECT idx, label FROM samples
                                 WHERE partition=? AND idx>=? AND idx<? ORDER BY idx""",
                              (partition, begin, end))
            for idx, label in rows:
                yield self._sample(partition, idx, label)

    def get_sample(self, sample_id: str) -> Sample:
        partition, _, suffix = sample_id.partition("/")
        if partition not in FILES or not suffix.isascii() or not suffix.isdecimal():
            raise KeyError(sample_id)
        if len(suffix) > len(str(len(self.images[partition]))) or str(int(suffix)) != suffix:
            raise KeyError(sample_id)
        with connect(self.index, read_only=True) as db:
            row = db.execute("SELECT idx, label FROM samples WHERE partition=? AND idx=?",
                             (partition, int(suffix))).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return self._sample(partition, *row)


class MnistDataset(CatalogDataset):
    @classmethod
    def resolve_settings(cls, settings: dict, root: Path) -> dict:
        path = settings.get("path", CORPUS.as_posix())
        if (settings.keys() - {"path", "metadata_sha256", "validation_size"}
                or not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts):
            raise ValueError("MNIST requires a project-relative corpus path")
        expected = settings.get("metadata_sha256")
        index = _index_path(root, expected)
        digest = _digest(index)
        if expected is not None and expected != digest:
            raise ValueError("MNIST metadata checksum mismatch")
        resources = _resources(index)
        size = settings.get("validation_size", 5000)
        if type(size) is not int or not 0 < size < resources[FILES["train"][0]][0]:
            raise ValueError("validation_size must leave nonempty training and validation splits")
        _check_corpus(root / path, resources)
        return {"path": path, "metadata_sha256": digest, "validation_size": size}

    @classmethod
    def replay_files(cls, settings: dict, root: Path) -> dict[str, Path]:
        # The frozen metadata records sample IDs, labels and checksums. Images stay external.
        digest = settings["metadata_sha256"]
        return {digest: _index_path(root, digest)}

    @classmethod
    def build(cls, settings: dict, *, root: Path) -> MnistDataset:
        settings = cls.resolve_settings(settings, root)
        digest = settings["metadata_sha256"]
        index = store_path(root) / "inputs" / digest
        if not index.is_file():
            # Experiment retains replay_files at acceptance. Support direct Dataset use too.
            index.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=index.parent) as stage:
                copy = Path(stage) / "metadata.sqlite3"
                shutil.copyfile(root / INDEX, copy)
                if _digest(copy) != digest:
                    raise ValueError("MNIST metadata changed during dataset creation")
                os.replace(copy, index)
        size = settings["validation_size"]
        version = _version(_resources(index), size)
        dataset = cls(_catalog(root), version)
        with connect(dataset.catalog) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY, metadata_sha256 TEXT NOT NULL, validation_size INTEGER NOT NULL
            )""")
            db.execute("INSERT OR IGNORE INTO versions VALUES (?, ?, ?)", (version, digest, size))
        dataset.root = root
        dataset.corpus = root / settings["path"]
        return dataset

    @staticmethod
    def _read_version(root: Path, corpus: Path, version_id: str) -> tuple[Path, dict, int]:
        check_digest(version_id)
        with connect(_catalog(root), read_only=True) as db:
            row = db.execute("SELECT metadata_sha256, validation_size FROM versions WHERE id=?",
                             (version_id,)).fetchone()
        if row is None:
            raise KeyError(version_id)
        digest, size = row
        check_digest(digest)
        index = store_path(root) / "inputs" / digest
        if _digest(index) != digest:
            raise ValueError("MNIST retained metadata checksum mismatch")
        resources = _resources(index)
        if _version(resources, size) != version_id:
            raise ValueError(f"MNIST dataset version checksum mismatch: {version_id}")
        _check_corpus(corpus, resources)
        return index, resources, size

    def open_version(self, version_id: str) -> MnistVersion:
        # Resolve the link once so moving it during a read cannot mix corpora.
        corpus = self.corpus.resolve(strict=True)
        index, resources, size = self._read_version(self.root, corpus, version_id)
        return MnistVersion(index, corpus, resources, size)

    @classmethod
    def verify_version(cls, settings: dict, root: Path, version_id: str) -> None:
        cls._read_version(root, root / settings.get("path", CORPUS), version_id)
