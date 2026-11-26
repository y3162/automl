"""Versioned XOR inputs, binary labels, fixed splits and feature scaling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from automl.dataset import DatasetVersion, Sample
from automl.paths import work_directory
from automl.transform import Transform, TransformResult
from .catalog import CatalogDataset, check_digest, connect, json_bytes, store_path


def _sample(line: bytes) -> Sample:
    try:
        sample_id, x, y, label = next(csv.reader([line.decode("utf-8")], strict=True))
        if not sample_id.isascii() or not sample_id.isdecimal() or str(int(sample_id)) != sample_id:
            raise ValueError("sample ID must be a canonical nonnegative integer")
        if label not in ("0", "1"):
            raise ValueError("expected a binary label")
        features = (float(x), float(y))
        if not all(math.isfinite(value) for value in features):
            raise ValueError("expected finite coordinates")
    except (ValueError, UnicodeError, csv.Error, StopIteration) as exc:
        raise ValueError(f"invalid XOR CSV row: {exc}") from exc
    return Sample(sample_id, features, {"label": int(label)})


def _scan(path: Path) -> tuple[str, list[tuple[str, str, int]]]:
    """Validate one immutable version and derive its index without retaining row payloads."""
    seen = set()
    rows = []
    with path.open("rb") as stream:
        header = stream.readline()
        if header.rstrip(b"\r\n") != b"sample_id,x1,x2,label":
            raise ValueError("expected XOR CSV header sample_id,x1,x2,label")
        digest = hashlib.sha256(header)
        offset = len(header)
        for line in stream:
            sample_id = _sample(line).sample_id
            if sample_id in seen:
                raise ValueError(f"duplicate XOR sample ID {sample_id}")
            seen.add(sample_id)
            # Each block is a quadrant of the fixture. IDs keep their split across revisions.
            position = int(sample_id) % 128
            split = "train" if position < 80 else "validation" if position < 104 else "test"
            rows.append((sample_id, split, offset))
            digest.update(line)
            offset += len(line)
    return digest.hexdigest(), rows


def _index_path(root: Path) -> Path:
    return work_directory(root, fallback=root / "work") / "xor" / "index.sqlite3"


class CsvVersion(DatasetVersion):
    def __init__(self, index: Path, version: str, raw: Path):
        self.catalog = index
        self.version = version
        self.raw = raw
        digest, expected = _scan(raw)
        if digest != version:
            raise ValueError(f"XOR snapshot checksum mismatch: {version}")
        index.parent.mkdir(parents=True, exist_ok=True)
        with connect(index) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS samples (
                    version TEXT, id TEXT, split TEXT, offset INTEGER, PRIMARY KEY (version, id)
                );
                CREATE INDEX IF NOT EXISTS by_split ON samples (version, split, offset);
            """)
            db.execute("BEGIN IMMEDIATE")
            actual = db.execute(
                "SELECT id, split, offset FROM samples WHERE version=? ORDER BY offset", (version,)
            ).fetchall()
            if actual != expected:
                db.execute("DELETE FROM samples WHERE version=?", (version,))
                db.executemany("INSERT INTO samples VALUES (?,?,?,?)", ((version, *row) for row in expected))

    @staticmethod
    def _read(stream: BinaryIO, offset: int, sample_id: str) -> Sample:
        stream.seek(offset)
        sample = _sample(stream.readline())
        if sample.sample_id != sample_id:
            raise ValueError(f"XOR index mismatch for {sample_id}")
        return sample

    def get_sample(self, sample_id: str) -> Sample:
        with connect(self.catalog) as db:
            found = db.execute(
                "SELECT offset FROM samples WHERE version=? AND id=?", (self.version, sample_id)
            ).fetchone()
        if found is None:
            raise KeyError(sample_id)
        with self.raw.open("rb") as stream:
            return self._read(stream, found[0], sample_id)

    def iter_samples(self, *, split: str) -> Iterator[Sample]:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"unknown XOR split: {split}")
        with connect(self.catalog) as db, self.raw.open("rb") as stream:
            for sample_id, offset in db.execute(
                "SELECT id, offset FROM samples WHERE version=? AND split=? ORDER BY offset", (self.version, split)
            ):
                yield self._read(stream, offset, sample_id)


class CsvDataset(CatalogDataset):
    @classmethod
    def verify_version(cls, settings: dict, root: Path, version_id: str) -> None:
        """Verify authoritative inputs without creating or repairing a work index."""
        check_digest(version_id)
        store = store_path(root)
        with connect(store / "datasets" / "xor" / "csv.sqlite3", read_only=True) as db:
            found = db.execute("SELECT 1 FROM versions WHERE id=?", (version_id,)).fetchone()
        if found is None:
            raise ValueError(f"unregistered XOR CSV version: {version_id}")
        digest, _ = _scan(store / "inputs" / version_id)
        if digest != version_id:
            raise ValueError(f"XOR CSV checksum mismatch: {version_id}")

    @classmethod
    def replay_files(cls, settings: dict, root: Path) -> dict[str, Path]:
        digest = settings["sha256"]
        check_digest(digest)
        retained = store_path(root) / "inputs" / digest
        return {digest: retained if retained.is_file() else root / settings["path"]}

    @classmethod
    def resolve_settings(cls, settings: dict, root: Path) -> dict:
        path = settings.get("path")
        if (settings.keys() - {"path", "sha256"} or not isinstance(path, str) or not path
                or Path(path).is_absolute() or ".." in Path(path).parts):
            raise ValueError("XOR CSV requires a project-relative path and optional sha256")
        specified = settings.get("sha256")
        candidate = root / path if specified is None else cls.replay_files(settings, root)[specified]
        digest, _ = _scan(candidate)
        if specified is not None and specified != digest:
            raise ValueError(f"XOR CSV checksum mismatch: {candidate}")
        return {"path": path, "sha256": digest}

    @classmethod
    def build(cls, settings: dict, *, root: Path) -> CsvDataset:
        resolved = cls.resolve_settings(settings, root)
        digest = resolved["sha256"]
        store = store_path(root)
        retained = store / "inputs" / digest
        if not retained.is_file():
            retained.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=retained.parent, delete=False) as file:
                temporary = Path(file.name)
                try:
                    with (root / resolved["path"]).open("rb") as source:
                        shutil.copyfileobj(source, file)
                    file.flush()
                    file.seek(0)
                    if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                        raise ValueError("XOR CSV changed before snapshot creation")
                    os.fsync(file.fileno())
                    os.replace(temporary, retained)
                finally:
                    temporary.unlink(missing_ok=True)
        instance = cls(store / "datasets" / "xor" / "csv.sqlite3", digest)
        with connect(instance.catalog) as db:
            db.execute("CREATE TABLE IF NOT EXISTS versions (id TEXT PRIMARY KEY)")
            db.execute("INSERT OR IGNORE INTO versions VALUES (?)", (digest,))
        instance.store = store
        return instance

    def open_version(self, version_id: str) -> CsvVersion:
        with connect(self.catalog) as db:
            found = db.execute("SELECT 1 FROM versions WHERE id=?", (version_id,)).fetchone()
        if found is None:
            raise KeyError(version_id)
        path = self.store / "inputs" / version_id
        return CsvVersion(_index_path(self.store), version_id, path)


class InlineVersion(DatasetVersion):
    def __init__(self, splits: dict):
        self._splits = splits

    @staticmethod
    def _sample(sample_id: str, row: list) -> Sample:
        x, y, label = row
        return Sample(sample_id, (x, y), {"label": label})

    def get_sample(self, sample_id: str) -> Sample:
        split, sep, position = sample_id.rpartition("/")
        if not sep or not position.isascii() or not position.isdecimal():
            raise KeyError(sample_id)
        i = int(position)
        if str(i) != position or split not in self._splits or i >= len(self._splits[split]):
            raise KeyError(sample_id)
        return self._sample(sample_id, self._splits[split][i])

    def iter_samples(self, *, split: str) -> Iterator[Sample]:
        if split not in self._splits:
            raise ValueError(f"unknown XOR split: {split}")
        for i, row in enumerate(self._splits[split]):
            yield self._sample(f"{split}/{i}", row)


class InlineDataset(CatalogDataset):
    @classmethod
    def verify_version(cls, settings: dict, root: Path, version_id: str) -> None:
        check_digest(version_id)
        catalog = store_path(root) / "datasets" / "xor" / "inline.sqlite3"
        with connect(catalog, read_only=True) as db:
            row = db.execute("SELECT splits FROM versions WHERE id=?", (version_id,)).fetchone()
        if row is None:
            raise ValueError(f"unregistered inline XOR version: {version_id}")
        if hashlib.sha256(row[0]).hexdigest() != version_id:
            raise ValueError(f"inline XOR data checksum mismatch: {version_id}")
        cls.resolve_settings({"splits": json.loads(row[0])}, root)

    @classmethod
    def resolve_settings(cls, settings: dict, root: Path) -> dict:
        splits = settings.get("splits")
        if settings.keys() != {"splits"} or not isinstance(splits, dict) or not splits:
            raise ValueError("inline XOR data requires a splits mapping")
        for split, rows in splits.items():
            if not isinstance(split, str) or not split or not isinstance(rows, list):
                raise ValueError("each named XOR split must contain a list of rows")
            for i, row in enumerate(rows):
                if (not isinstance(row, list) or len(row) != 3
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in row[:2])
                        or type(row[2]) is not int or row[2] not in (0, 1)):
                    raise ValueError(f"invalid XOR sample {split}/{i}: expected [x1, x2, binary label]")
        return settings

    @classmethod
    def build(cls, settings: dict, *, root: Path) -> InlineDataset:
        payload = json_bytes(cls.resolve_settings(settings, root)["splits"])
        version = hashlib.sha256(payload).hexdigest()
        instance = cls(store_path(root) / "datasets" / "xor" / "inline.sqlite3", version)
        with connect(instance.catalog) as db:
            db.execute("CREATE TABLE IF NOT EXISTS versions (id TEXT PRIMARY KEY, splits BLOB NOT NULL)")
            db.execute("INSERT OR IGNORE INTO versions VALUES (?,?)", (version, payload))
        return instance

    def open_version(self, version_id: str) -> InlineVersion:
        with connect(self.catalog) as db:
            found = db.execute("SELECT splits FROM versions WHERE id=?", (version_id,)).fetchone()
        if found is None:
            raise KeyError(version_id)
        return InlineVersion(json.loads(found[0]))


class ScaleFeatures(Transform):
    def __init__(self, scale: float = 0.5) -> None:
        self.scale = scale

    @classmethod
    def resolve_settings(cls, settings: dict, root: Path) -> dict:
        resolved = super().resolve_settings(settings, root)
        scale = resolved["scale"]
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale == 0:
            raise ValueError("scale must be finite and nonzero")
        return resolved

    def apply(self, data: Any, *, reproduction: Any = None) -> TransformResult:
        applied = {"scale": self.scale}
        if reproduction is not None and reproduction != applied:
            raise ValueError("scale differs from the recorded reproduction conditions")
        return TransformResult(np.asarray(data, dtype=float) * self.scale, applied)
