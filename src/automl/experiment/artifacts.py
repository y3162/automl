"""Persistent, content-verified run artifacts and checkpoint leases."""

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from errno import ENOTEMPTY
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from .errors import ExperimentError


ARTIFACT_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL,
    details TEXT NOT NULL,
    files TEXT NOT NULL,
    training_order INTEGER,
    checkpoint_state TEXT,
    CHECK (
        (kind = 'checkpoint' AND training_order IS NOT NULL
           AND checkpoint_state IS NOT NULL
           AND checkpoint_state IN ('retained', 'deleting', 'deleted'))
        OR (kind != 'checkpoint' AND checkpoint_state IS NULL)
    )
);
CREATE INDEX IF NOT EXISTS artifact_run_order
    ON artifacts(run_id, kind, checkpoint_state, training_order);
CREATE TABLE IF NOT EXISTS checkpoint_users (
    checkpoint_id TEXT NOT NULL REFERENCES artifacts(id),
    user TEXT NOT NULL,
    PRIMARY KEY (checkpoint_id, user)
);
"""


def _record(row):
    try:
        details = json.loads(row["details"])
        files = json.loads(row["files"])
    except (ValueError, TypeError) as exc:
        raise ExperimentError("invalid_request", "Unreadable artifact record", artifact_id=row["id"]) from exc
    if not isinstance(files, dict) or not files or not isinstance(details, dict):
        raise ExperimentError("invalid_request", "Missing artifact file index or details", artifact_id=row["id"])
    return {
        "id": row["id"], "run_id": row["run_id"], "kind": row["kind"],
        "details": details, "files": files,
        "order": row["training_order"], "checkpoint_state": row["checkpoint_state"],
    }


def _filename(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ExperimentError("invalid_request", "Invalid artifact filename or identifier", name=name)
    return name


def _copy_content(content, destination):
    """Hash exactly the bytes copied, even if an external input changes later."""
    digest = hashlib.sha256()
    size = 0
    if isinstance(content, (bytes, bytearray, memoryview)):
        source = BytesIO(content)
    else:
        source = Path(content).open("rb")
    with source as inp, destination.open("xb") as out:
        while block := inp.read(1 << 20):
            out.write(block)
            digest.update(block)
            size += len(block)
        out.flush()
        os.fsync(out.fileno())
    return {"size": size, "sha256": digest.hexdigest()}


class Artifacts:
    def __init__(self, store):
        self.store = store
        self.root = Path(store.root).resolve()

    def save(self, db, run_id, kind, files, *, details=None, order=None, artifact_id=None):
        """Store files and publish them in the caller's transaction."""
        with self.stage(run_id, kind, files, details=details, order=order, artifact_id=artifact_id) as publish:
            return publish(db)

    @contextmanager
    def stage(self, run_id, kind, files, *, details=None, order=None, artifact_id=None):
        """Write privately and yield a function to publish in a short transaction.

        Abandoned attempts leave no visible artifact. A losing concurrent
        writer discards its private directory without touching the winner.
        """
        if not files:
            raise ExperimentError("invalid_request", "An artifact needs at least one file", kind=kind)
        if kind == "checkpoint" and (not isinstance(order, int) or isinstance(order, bool)):
            raise ExperimentError("invalid_request", "Checkpoint needs an integer training order", order=order)
        _filename(run_id)
        artifact_id = _filename(artifact_id or f"{'C' if kind == 'checkpoint' else 'A'}{uuid4().hex}")
        if details is not None and not isinstance(details, dict):
            raise ExperimentError("invalid_request", "Artifact details must be a dictionary", kind=kind)
        metadata = json.dumps(details or {}, ensure_ascii=False, allow_nan=False)
        base = self.root / "runs" / run_id / "artifacts"
        if base.resolve() != base:
            raise ExperimentError("invalid_request", "Artifact directory has moved", run_id=run_id)
        base.mkdir(parents=True, exist_ok=True)
        directory = Path(mkdtemp(dir=base, prefix=f"{artifact_id}-"))
        index = {}
        published = False

        def publish(db):
            nonlocal published
            if db.execute("SELECT 1 FROM artifacts WHERE id = ?", (artifact_id,)).fetchone():
                raise ExperimentError("invalid_request", "Artifact ID already exists", artifact_id=artifact_id)
            if db.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise ExperimentError("not_found", "Unknown artifact owner", run_id=run_id)
            db.execute(
                "INSERT INTO artifacts(id,run_id,kind,details,files,training_order,checkpoint_state)"
                " VALUES(?,?,?,?,?,?,?)",
                (artifact_id, run_id, kind, metadata, json.dumps(index), order,
                 "retained" if kind == "checkpoint" else None),
            )
            self.store.touch(db, run_id)
            self.store.on_rollback(db, lambda: shutil.rmtree(directory, ignore_errors=True))
            published = True
            return artifact_id

        try:
            for name, content in files.items():
                filename = _filename(name)
                if kind == "checkpoint":
                    filename = f"step_{order}_{filename}"
                path = directory / filename
                entry = _copy_content(content, path)
                entry["path"] = path.relative_to(self.root).as_posix()
                index[name] = entry
            yield publish
        finally:
            if not published:
                shutil.rmtree(directory, ignore_errors=True)

    @staticmethod
    def get(db, artifact_id):
        row = db.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise ExperimentError("not_found", "Artifact not found", artifact_id=artifact_id)
        return _record(row)

    @staticmethod
    def _rows(db, run_id):
        if run_id is None:
            return db.execute("SELECT * FROM artifacts ORDER BY rowid")
        return db.execute("SELECT * FROM artifacts WHERE run_id = ? ORDER BY rowid", (run_id,))

    def list(self, db, run_id=None):
        return [_record(row) for row in self._rows(db, run_id)]

    @staticmethod
    def last_checkpoint(db, run_id):
        """The last published checkpoint, irrespective of its retention state."""
        row = db.execute("SELECT * FROM artifacts WHERE run_id=? AND kind='checkpoint' ORDER BY rowid DESC LIMIT 1",
                         (run_id,)).fetchone()
        return _record(row) if row else None

    @staticmethod
    def _retained(item):
        if item["kind"] != "checkpoint":
            raise ExperimentError("not_found", "Expected a checkpoint", checkpoint_id=item["id"])
        if item["checkpoint_state"] != "retained":
            raise ExperimentError("invalid_request", "Checkpoint is unavailable",
                                  checkpoint_id=item["id"], checkpoint_state=item["checkpoint_state"])

    def resolve(self, db, ref):
        """Resolve a checkpoint ID, run ID, or RUN:latest to a usable checkpoint."""
        if not isinstance(ref, str) or not ref:
            raise ExperimentError("invalid_request", "Expected a checkpoint reference", ref=ref)
        row = db.execute("SELECT * FROM artifacts WHERE id = ?", (ref,)).fetchone()
        if row is not None:
            item = _record(row)
            self._retained(item)
            for name in item["files"]:
                self._checked_file(item, name)
            return item
        run = ref.removesuffix(":latest")
        if db.execute("SELECT 1 FROM runs WHERE id = ?", (run,)).fetchone() is None:
            raise ExperimentError("not_found", "Checkpoint reference not found", ref=ref)
        candidates = db.execute(
            "SELECT * FROM artifacts WHERE run_id = ? AND kind = 'checkpoint'"
            " AND checkpoint_state = 'retained' ORDER BY training_order DESC, rowid DESC", (run,),
        )
        issues = []
        for row in candidates:
            try:
                item = _record(row)
                for name in item["files"]:
                    self._checked_file(item, name)
            except ExperimentError as exc:
                issues.append({"checkpoint_id": row["id"], "reason": str(exc)})
            else:
                return item
        raise ExperimentError("invalid_request", "No usable checkpoint remains", run_id=run, issues=issues)

    def acquire(self, db, checkpoint_id, user):
        """Pin a retained checkpoint within the transaction accepting its use."""
        self._retained(self.get(db, checkpoint_id))
        db.execute("INSERT OR IGNORE INTO checkpoint_users(checkpoint_id,user) VALUES (?,?)",
                   (checkpoint_id, user))

    @staticmethod
    def release(db, user, checkpoint_id=None):
        if checkpoint_id is None:
            db.execute("DELETE FROM checkpoint_users WHERE user = ?", (user,))
        else:
            db.execute("DELETE FROM checkpoint_users WHERE user = ? AND checkpoint_id = ?",
                       (user, checkpoint_id))

    @staticmethod
    def release_run(db, run_id):
        """Release start-state and per-evaluation leases after confirmed process exit."""
        db.execute("DELETE FROM checkpoint_users WHERE user = ? OR instr(user, ?) = 1",
                   (run_id, run_id + ":"))

    def _location(self, item, name):
        if name not in item["files"]:
            raise ExperimentError("not_found", "File not found in artifact", artifact_id=item["id"], name=name)
        entry = item["files"][name]
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str) or not isinstance(entry.get("size"), int)):
            raise ExperimentError("invalid_request", "Incomplete artifact file entry", artifact_id=item["id"],
                                  name=name)
        relative = Path(entry["path"])
        base = self.root / "runs" / _filename(item["run_id"]) / "artifacts"
        path = self.root / relative
        if (relative.is_absolute() or "\x00" in str(relative) or ".." in relative.parts or path.parent.parent != base
                or not path.parent.name.startswith(item["id"] + "-") or path.resolve() != path):
            raise ExperimentError("invalid_request", "Invalid stored artifact path", artifact_id=item["id"],
                                  path=str(relative))
        return path

    def _checked_file(self, item, name, *, load=False):
        if item["kind"] == "checkpoint":
            self._retained(item)
        path = self._location(item, name)
        entry = item["files"][name]
        try:
            with path.open("rb") as file:
                if load:
                    data = file.read()
                    digest = hashlib.sha256(data).hexdigest()
                else:
                    digest = hashlib.file_digest(file, "sha256").hexdigest()
                size = os.fstat(file.fileno()).st_size
        except OSError as exc:
            raise ExperimentError("invalid_request", f"Cannot read artifact file: {exc}",
                                  artifact_id=item["id"], path=entry["path"]) from exc
        if digest != entry["sha256"]:
            raise ExperimentError("invalid_request", "Artifact SHA-256 mismatch", artifact_id=item["id"],
                                  path=entry["path"], expected=entry["sha256"], actual=digest)
        if size != entry["size"]:
            raise ExperimentError("invalid_request", "Artifact file size mismatch", artifact_id=item["id"],
                                  path=entry["path"])
        return data if load else path

    def path(self, db, artifact_id, name):
        return self._checked_file(self.get(db, artifact_id), name)

    def read(self, db, artifact_id, name):
        return self._checked_file(self.get(db, artifact_id), name, load=True)

    def delete(self, checkpoint_id):
        # Commit the intent before the first unlink. Concurrent users and crash
        # recovery both see the same state even if only half the files are gone.
        with self.store.transaction(write=True) as db:
            item = self.get(db, checkpoint_id)
            if item["kind"] != "checkpoint":
                raise ExperimentError("not_found", "Expected a checkpoint", checkpoint_id=checkpoint_id)
            if item["checkpoint_state"] == "deleted":
                return {"checkpoint_id": checkpoint_id, "result": "deleted"}
            if item["checkpoint_state"] == "retained":
                in_use = db.execute("SELECT 1 FROM checkpoint_users WHERE checkpoint_id = ? LIMIT 1",
                                    (checkpoint_id,)).fetchone()
                if in_use:
                    return {"checkpoint_id": checkpoint_id, "result": "in_use"}
                db.execute("UPDATE artifacts SET checkpoint_state = 'deleting' WHERE id = ?", (checkpoint_id,))
                self.store.touch(db, item["run_id"])
        try:
            directories = set()
            for name in item["files"]:
                path = self._location(item, name)
                directories.add(path.parent)
                path.unlink(missing_ok=True)
            for directory in directories:
                try:
                    directory.rmdir()
                except FileNotFoundError:
                    pass  # Another caller may have finished this deletion.
                except OSError as exc:
                    if exc.errno != ENOTEMPTY:
                        raise
        except OSError as exc:
            raise ExperimentError("invalid_request", f"Checkpoint deletion incomplete: {exc}",
                                  checkpoint_id=checkpoint_id, checkpoint_state="deleting") from exc
        with self.store.transaction(write=True) as db:
            updated = db.execute("UPDATE artifacts SET checkpoint_state = 'deleted'"
                                 " WHERE id = ? AND checkpoint_state = 'deleting'", (checkpoint_id,))
            if updated.rowcount:
                self.store.touch(db, item["run_id"])
        return {"checkpoint_id": checkpoint_id, "result": "deleted"}

    def verify(self, db, run_id=None):
        report = {"checked": 0, "issues": [], "pending_deletions": []}
        for row in self._rows(db, run_id):
            if row["checkpoint_state"] == "deleting":
                report["pending_deletions"].append(row["id"])
            try:
                item = _record(row)
            except ExperimentError as exc:
                report["issues"].append({"run_id": row["run_id"], "artifact_id": row["id"],
                                         "reason": str(exc)})
                continue
            if item["checkpoint_state"] == "deleting":
                continue
            for name in item["files"]:
                report["checked"] += 1
                try:
                    if item["checkpoint_state"] == "deleted":
                        path = self._location(item, name)
                        if path.exists():
                            raise ValueError("Deleted checkpoint file still exists")
                    else:
                        self._checked_file(item, name)
                except (ExperimentError, OSError, ValueError, KeyError) as exc:
                    report["issues"].append({"run_id": item["run_id"], "artifact_id": item["id"],
                                             "name": name, "reason": str(exc)})
        return report
