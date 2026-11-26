"""Durable version requests and generated-sample recipes shared by the XOR datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from automl.dataset import Dataset


def json_bytes(value: Any) -> bytes:
    """XOR recipe hashes cover sorted-key UTF-8 JSON followed by one newline."""
    def array_value(item):
        if isinstance(item, (np.ndarray, np.generic)):
            return item.tolist()
        raise TypeError(f"{type(item).__name__} is not an XOR JSON value")

    return (json.dumps(value, sort_keys=True, allow_nan=False, default=array_value) + "\n").encode()


def store_path(root: Path) -> Path:
    return Path(os.environ.get("AUTOML_STORE", root / "data" / "store"))


@contextmanager
def connect(path: Path, *, read_only: bool = False):
    location = f"{path.absolute().as_uri()}?mode=ro" if read_only else path
    db = sqlite3.connect(location, uri=read_only, timeout=30)
    try:
        with db:
            yield db
    finally:
        db.close()


def check_digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")


def _check_request(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("a nonempty request ID is required")


class CatalogDataset(Dataset):
    def __init__(self, catalog: Path, version: str) -> None:
        self.catalog = catalog
        self.version = version
        catalog.parent.mkdir(parents=True, exist_ok=True)
        with connect(catalog) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generated (
                    request TEXT PRIMARY KEY, id TEXT UNIQUE,
                    registration BLOB, recipe BLOB, error TEXT
                );
            """)

    def create_version(self, request_id: str) -> str:
        _check_request(request_id)
        with connect(self.catalog) as db:
            db.execute("INSERT OR IGNORE INTO requests VALUES (?, ?)", (request_id, self.version))
            return db.execute("SELECT version FROM requests WHERE id=?", (request_id,)).fetchone()[0]

    def _check_recipe(self, recipe: dict, content: Any) -> None:
        if not isinstance(recipe, dict):
            raise ValueError("an XOR recipe must be a mapping")
        if "input" in recipe:
            ref = recipe["input"]
            if not isinstance(ref, (list, tuple)) or len(ref) != 2 or not all(isinstance(i, str) for i in ref):
                raise ValueError("recipe input must contain [version_id, sample_id]")
            try:
                self.open_version(ref[0]).get_sample(ref[1])
            except (KeyError, FileNotFoundError) as exc:
                raise ValueError(f"recipe input unavailable: {ref}") from exc
        if "expected_sha256" in recipe:
            digest = recipe["expected_sha256"]
            check_digest(digest)
            if content is not None and hashlib.sha256(json_bytes(content)).hexdigest() != digest:
                raise ValueError("generated content differs from recipe expected_sha256")

    def register_generated(self, request_id: str, *, recipe: Any = None, content: Any = None) -> str:
        _check_request(request_id)
        # Persist the pending request before validation so an interrupted caller may retry it.
        with connect(self.catalog) as db:
            db.execute("INSERT OR IGNORE INTO generated (request) VALUES (?)", (request_id,))
        with connect(self.catalog) as db:
            db.execute("BEGIN IMMEDIATE")
            sample_id, saved = db.execute(
                "SELECT id, registration FROM generated WHERE request=?", (request_id,)
            ).fetchone()
            try:
                payload = json_bytes({"recipe": recipe, "content": content})
                if sample_id is not None:
                    if payload != saved:
                        raise ValueError(f"registration {request_id} has different content")
                    return sample_id
                if recipe is None and content is None:
                    raise ValueError("registration needs a recipe or content")
                if recipe is not None:
                    self._check_recipe(recipe, content)
            except (TypeError, ValueError, OSError) as exc:
                if sample_id is None:
                    db.execute("UPDATE generated SET error=? WHERE request=?", (str(exc), request_id))
                    db.commit()
                raise
            sample_id = uuid4().hex
            db.execute(
                "UPDATE generated SET id=?, registration=?, recipe=?, error=NULL WHERE request=?",
                (sample_id, payload, json_bytes(recipe), request_id),
            )
            return sample_id

    def registration_status(self, request_id: str) -> dict:
        with connect(self.catalog) as db:
            row = db.execute("SELECT id, error FROM generated WHERE request=?", (request_id,)).fetchone()
        if row is None:
            return {"status": "unknown"}
        if row[0] is not None:
            return {"status": "succeeded", "sample_id": row[0]}
        if row[1] is not None:
            return {"status": "failed", "reason": row[1]}
        return {"status": "pending"}

    def get_generated(self, sample_id: str) -> dict:
        with connect(self.catalog) as db:
            row = db.execute("SELECT registration, recipe FROM generated WHERE id=?", (sample_id,)).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return {"recipe": json.loads(row[1]), "content": json.loads(row[0])["content"]}

    def ensure_recipe(self, sample_id: str, *, recipe: dict | None = None) -> dict:
        """Require a versioned input, transform version and applied conditions for training.

        A content-only registration can supply its recipe on first use. Further uses
        read the same recipe, and retrying the original registration keeps its ID.
        """
        with connect(self.catalog) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT registration, recipe FROM generated WHERE id=?", (sample_id,)).fetchone()
            if row is None:
                raise KeyError(sample_id)
            stored = json.loads(row[1])
            if stored is not None:
                if recipe is not None and json_bytes(recipe) != row[1]:
                    raise ValueError(f"generated sample {sample_id} already has a different recipe")
                recipe = stored
            if (not isinstance(recipe, dict) or not recipe.get("input")
                    or not isinstance(recipe.get("transform"), str) or not recipe["transform"]
                    or not isinstance(recipe.get("transform_version"), str) or not recipe["transform_version"]
                    or not isinstance(recipe.get("conditions"), dict)):
                raise ValueError("training requires a recipe with input, transform, transform_version and conditions")
            self._check_recipe(recipe, json.loads(row[0])["content"])
            if stored is None:
                db.execute("UPDATE generated SET recipe=? WHERE id=?", (json_bytes(recipe), sample_id))
            return recipe

    def verify_generated(self, sample_id: str, regenerated: Any) -> bool:
        saved = self.get_generated(sample_id)
        actual = json_bytes(regenerated)
        if saved["content"] is not None:
            return actual == json_bytes(saved["content"])
        expected = saved["recipe"].get("expected_sha256")
        if expected is None:
            raise ValueError(f"generated sample {sample_id} has no comparison reference")
        return hashlib.sha256(actual).hexdigest() == expected
