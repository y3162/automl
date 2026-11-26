"""SQLite transactions for experiment admission, process state, and portable history."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from automl.dataset.usage import USAGE_SCHEMA, read_usage, verify_commits
from .artifacts import ARTIFACT_SCHEMA, Artifacts
from .errors import ExperimentError
from .evaluations import EVALUATION_SCHEMA, evaluation_summary, fail_open_evaluations


ACTIVE = ("accepted", "running")
TERMINAL = ("succeeded", "failed", "interrupted")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_identity (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    project_id TEXT NOT NULL
);
INSERT OR IGNORE INTO store_identity VALUES(1,lower(hex(randomblob(16))));
CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    origin_run TEXT REFERENCES runs(id),
    origin_checkpoint TEXT REFERENCES artifacts(id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    branch TEXT NOT NULL REFERENCES branches(id),
    sequence INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('accepted','running','succeeded','failed','interrupted')),
    record TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    UNIQUE(branch, sequence)
);
CREATE UNIQUE INDEX IF NOT EXISTS occupied_branch ON runs(branch)
    WHERE status IN ('accepted','running');
CREATE TABLE IF NOT EXISTS requests (
    scope TEXT NOT NULL,
    request_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    args TEXT NOT NULL,
    result TEXT NOT NULL,
    PRIMARY KEY(scope, request_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS run_events ON events(run_id,id);
"""

_USAGE_REVISIONS = """
CREATE TRIGGER IF NOT EXISTS usage_revision AFTER INSERT ON dataset_usage
BEGIN UPDATE runs SET revision=revision+1 WHERE id=NEW.run_id; END;
CREATE TRIGGER IF NOT EXISTS usage_commit_revision AFTER INSERT ON dataset_usage_commits
BEGIN UPDATE runs SET revision=revision+1 WHERE id=NEW.run_id; END;
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    # A path and its string representation identify the same API argument.
    def default(item):
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot store an object of type {type(item).__name__}")

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=default, allow_nan=False)


class _Connection(sqlite3.Connection):
    """Keep cleanup actions private to the transaction that created its files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollback_callbacks = []


class Store:
    def __init__(self, root, *, read_only=False):
        self.root = Path(root).resolve()
        self.read_only = read_only
        self.path = self.root / "history.sqlite3"
        self.artifacts = Artifacts(self)
        if read_only:
            if not self.path.is_file():
                raise ExperimentError("not_found", "History database does not exist", root=str(self.root))
            return
        self.root.mkdir(parents=True, exist_ok=True)
        db = self._connect()
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] < 2:
                # Competing first writers serialize at BEGIN IMMEDIATE. Every DDL
                # statement is idempotent, including creation of the revision triggers.
                db.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA + ARTIFACT_SCHEMA
                                 + USAGE_SCHEMA + EVALUATION_SCHEMA + _USAGE_REVISIONS
                                 + "PRAGMA user_version=2; COMMIT;")
        finally:
            db.close()

    def _connect(self):
        uri = self.path.as_uri() + ("?mode=ro" if self.read_only else "?mode=rwc")
        db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=60, factory=_Connection)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @property
    def project_id(self):
        with self.transaction() as db:
            return db.execute("SELECT project_id FROM store_identity").fetchone()[0]

    @staticmethod
    def on_rollback(db, callback):
        db.rollback_callbacks.append(callback)

    @contextmanager
    def transaction(self, write=False):
        if write and self.read_only:
            raise ExperimentError("invalid_request", "The experiment was opened read-only")
        db = self._connect()
        try:
            if not write:
                db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            for callback in reversed(db.rollback_callbacks):
                callback()
            raise
        finally:
            db.close()

    @staticmethod
    def request(db, request_id, operation, args, *, scope="project"):
        if request_id is None:
            return None
        if not isinstance(request_id, str) or not request_id:
            raise ExperimentError("invalid_request", "Request ID must be a nonempty string")
        row = db.execute("SELECT operation,args,result FROM requests WHERE scope=? AND request_id=?",
                         (scope, request_id)).fetchone()
        if row is None:
            return None
        if (operation, _json(args)) != (row["operation"], row["args"]):
            raise ExperimentError("invalid_request", "Request ID already used with different arguments",
                                  request_id=request_id, original_operation=row["operation"],
                                  original_args=json.loads(row["args"]), operation=operation, args=args)
        return json.loads(row["result"])

    @staticmethod
    def receipt(db, request_id, operation, args, result, *, scope="project"):
        if request_id is not None:
            db.execute("INSERT INTO requests(scope,request_id,operation,args,result) VALUES(?,?,?,?,?)",
                       (scope, request_id, operation, _json(args), _json(result)))

    def create_branch(self, db, *, origin_run=None, origin_checkpoint=None):
        if origin_checkpoint is not None:
            checkpoint = self.artifacts.resolve(db, origin_checkpoint)
            if origin_run is not None and checkpoint["run_id"] != origin_run:
                raise ExperimentError("invalid_request", "Checkpoint belongs to a different run",
                                      origin_run=origin_run, checkpoint_id=checkpoint["id"])
            origin_run, origin_checkpoint = checkpoint["run_id"], checkpoint["id"]
        if origin_run is not None:
            self.run(db, origin_run)
        branch_id = "B" + uuid4().hex
        db.execute("INSERT INTO branches(id,origin_run,origin_checkpoint,created_at) VALUES(?,?,?,?)",
                   (branch_id, origin_run, origin_checkpoint, _now()))
        return branch_id

    @staticmethod
    def branch(db, branch_id):
        row = db.execute("SELECT b.*,r.branch AS origin_branch FROM branches b "
                         "LEFT JOIN runs r ON b.origin_run=r.id WHERE b.id=?", (branch_id,)).fetchone()
        if row is None:
            raise ExperimentError("not_found", "Branch not found", branch_id=branch_id)
        return dict(row)

    @staticmethod
    def active_run(db, branch):
        row = db.execute("SELECT id FROM runs WHERE branch=? AND status IN ('accepted','running')",
                         (branch,)).fetchone()
        return row["id"] if row else None

    @staticmethod
    def last_training(db, branch):
        row = db.execute("SELECT record FROM runs WHERE branch=? "
                         "AND json_extract(record,'$.training_started')=1 ORDER BY sequence DESC LIMIT 1",
                         (branch,)).fetchone()
        return json.loads(row["record"]) if row else None

    def create_run(self, db, branch, config, *, run_id=None, **fields):
        parent = self.branch(db, branch)
        active = self.active_run(db, branch)
        if active:
            raise ExperimentError("branch_busy", "There is already an active run on this branch",
                                  branch=branch, active_run_id=active)
        run_id = run_id or "R" + uuid4().hex
        sequence = db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM runs WHERE branch=?",
                              (branch,)).fetchone()[0]
        record = {
            "origin_run": parent["origin_run"], "origin_checkpoint": parent["origin_checkpoint"],
            **fields, "id": run_id, "branch": branch, "sequence": sequence,
            "config": config, "created_at": _now(), "started_at": None, "ended_at": None,
            "status": "accepted", "state_confirmed": True, "training_started": False,
            "stop_requested": False, "stages": dict.fromkeys(config["plan"], "pending"),
            "stage_reasons": {},
        }
        db.execute("INSERT INTO runs(id,branch,sequence,status,record) VALUES(?,?,?,?,?)",
                   (run_id, branch, sequence, "accepted", _json(record)))
        self.event(db, run_id, "run_accepted", {"sequence": sequence})
        return run_id

    @staticmethod
    def run(db, run_id):
        row = db.execute("SELECT record FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ExperimentError("not_found", "Run not found", run_id=run_id)
        return json.loads(row["record"])

    @staticmethod
    def touch(db, run_id):
        db.execute("UPDATE runs SET revision=revision+1 WHERE id=?", (run_id,))

    def event(self, db, run_id, kind, payload):
        result = db.execute("INSERT INTO events(run_id,kind,payload,created_at) VALUES(?,?,?,?)",
                            (run_id, kind, _json(payload), _now()))
        self.touch(db, run_id)
        return result.lastrowid

    def update_run(self, db, run_id, **fields):
        record = self.run(db, run_id)
        if "status" in fields and record["status"] in TERMINAL and fields["status"] != record["status"]:
            raise ExperimentError("invalid_request", "Run is already final", run_id=run_id)
        if record["status"] == "accepted" and fields.get("status") == "running":
            fields.setdefault("started_at", _now())
        record.update(fields)
        db.execute("UPDATE runs SET status=?,record=?,revision=revision+1 WHERE id=?",
                   (record["status"], _json(record), run_id))
        return record

    def stage(self, db, run_id, stage, status, reason=None):
        record = self.run(db, run_id)
        if stage not in record["stages"]:
            raise ExperimentError("invalid_request", "Stage is not in the execution plan",
                                  run_id=run_id, stage=stage)
        previous = record["stages"][stage]
        if previous == status:
            return
        if previous in (*TERMINAL, "skipped"):
            raise ExperimentError("invalid_request", "Stage is already final",
                                  run_id=run_id, stage=stage, status=previous)
        record["stages"][stage] = status
        if reason is not None:
            record["stage_reasons"][stage] = reason
        fields = {"stages": record["stages"], "stage_reasons": record["stage_reasons"]}
        if stage == "train" and status == "running":
            fields["training_started"] = True
        self.update_run(db, run_id, **fields)
        self.event(db, run_id, "stage_changed", {"stage": stage, "status": status, "reason": reason})

    def finish(self, db, run_id, status, *, error=None):
        record = self.run(db, run_id)
        if record["status"] in TERMINAL:
            return record
        if status not in TERMINAL:
            raise ValueError(f"Unknown terminal status {status!r}")
        # A trainer may continue after an optional validation failed to
        # publish its report. Its worker can no longer produce any results.
        if status == "interrupted":
            reason = "Run interrupted"
            evaluation_state = "interrupted"
        else:
            reason = (error or {}).get("message", "Worker exited before reporting evaluation completion")
            evaluation_state = "failed"
        fail_open_evaluations(self, db, run_id, reason, state=evaluation_state)
        self.artifacts.release_run(db, run_id)
        fields = {"status": status, "ended_at": _now(), "state_confirmed": True, "error": error}
        if record["stages"].get("train") == "interrupted" and record.get("trainer_supports_resume") is False:
            fields["resume_reason"] = "The trainer does not support resuming training"
        elif record["stages"].get("train") == "interrupted":
            try:
                checkpoint = self.artifacts.resolve(db, run_id)
            except ExperimentError as exc:
                fields["resume_reason"] = exc.message
            else:
                fields["resume_checkpoint_available"] = checkpoint["id"]
        self.event(db, run_id, "run_finished", {"status": status, "error": error})
        return self.update_run(db, run_id, **fields)

    def status(self, run_id):
        with self.transaction() as db:
            return self.run(db, run_id)

    @staticmethod
    def _events(db, run_id=None):
        where = "" if run_id is None else " WHERE run_id=?"
        rows = db.execute("SELECT * FROM events" + where + " ORDER BY id",
                          () if run_id is None else (run_id,))
        return [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]

    def _history(self, db, run_id, summary):
        if run_id is None:
            runs = [json.loads(row[0]) for row in db.execute("SELECT record FROM runs ORDER BY rowid")]
            branches = [dict(row) for row in db.execute(
                "SELECT b.*,r.branch AS origin_branch FROM branches b "
                "LEFT JOIN runs r ON b.origin_run=r.id ORDER BY b.rowid")]
        else:
            runs = [self.run(db, run_id)]
            branches = [self.branch(db, runs[0]["branch"])]
        if summary:
            keys = ("id", "branch", "sequence", "status", "origin_run", "origin_checkpoint",
                    "resume_from", "resume_checkpoint", "initial_checkpoint", "stages", "stage_reasons")
            return {"branches": branches, "runs": [
                {key: run.get(key) for key in keys} | {"evaluations": evaluation_summary(db, run["id"])}
                for run in runs
            ]}
        result = {"branches": branches, "runs": runs, "events": self._events(db, run_id),
                  "artifacts": self.artifacts.list(db, run_id)}
        if run_id is not None:
            result["usage"] = read_usage(db, run_id)
        return result

    def history(self, run=None, *, summary=False):
        with self.transaction() as db:
            return self._history(db, run, summary)

    def tracking_revisions(self):
        with self.transaction() as db:
            return dict(db.execute("SELECT id,revision FROM runs ORDER BY rowid"))

    def tracking_snapshot(self, run):
        with self.transaction() as db:
            history = self._history(db, run, False)
            history["revision"] = db.execute("SELECT revision FROM runs WHERE id=?", (run,)).fetchone()[0]
            return history

    def _verify_run(self, db, record, report):
        run_id = record["id"]

        def issue(reason, **fields):
            report["issues"].append({"run_id": run_id, "reason": reason, **fields})

        def artifact_reference(field, artifact_id, *, kind=None, owner=None):
            if not artifact_id:
                return
            report["checked"] += 1
            row = db.execute("SELECT kind,run_id FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
            if row is None:
                issue("Referenced artifact has no record", item=field, artifact_id=artifact_id)
            elif ((kind is not None and row["kind"] != kind)
                  or (owner is not None and row["run_id"] != owner)):
                issue("Referenced artifact has a different kind or owner", item=field,
                      artifact_id=artifact_id, expected_kind=kind, expected_run=owner,
                      actual_kind=row["kind"], actual_run=row["run_id"])

        branch = db.execute("SELECT * FROM branches WHERE id=?", (record["branch"],)).fetchone()
        if branch is None:
            issue("Branch has no record", branch=record["branch"])
        else:
            for key in ("origin_run", "origin_checkpoint"):
                if record.get(key) != branch[key]:
                    issue("Run origin disagrees with the branch", item=key)
        for key in ("origin_run", "resume_from", "prediction_run"):
            target = record.get(key)
            if target and not db.execute("SELECT 1 FROM runs WHERE id=?", (target,)).fetchone():
                issue("Source run has no record", item=key, source_run=target)
        artifact_reference("origin_checkpoint", record.get("origin_checkpoint"), kind="checkpoint",
                           owner=record.get("origin_run"))
        artifact_reference("initial_checkpoint", record.get("initial_checkpoint"), kind="checkpoint",
                           owner=record.get("resume_from"))
        artifact_reference("prediction_artifact", record.get("prediction_artifact"),
                           owner=record.get("prediction_run"))
        for field in ("input_checkpoint_artifact", "input_predictions_artifact"):
            artifact_reference(field, record.get(field), owner=run_id)

        # Source files may have been intentionally deleted. The independent
        # start state and the order of its publication are what this run needs.
        saved, started = set(), set()
        for row in db.execute("SELECT id FROM events WHERE run_id=? AND NOT json_valid(payload)", (run_id,)):
            issue("Event payload is unreadable", event_id=row["id"])
        rows = db.execute("SELECT id,kind,payload FROM events WHERE run_id=? AND json_valid(payload) AND "
                          "kind IN ('stage_changed','start_state_saved') ORDER BY id", (run_id,))
        for row in rows:
            data = json.loads(row["payload"])
            if not isinstance(data, dict):
                issue("Invalid stage event", event_id=row["id"])
                continue
            if row["kind"] == "start_state_saved":
                stage = data.get("stage") or (
                    "train" if "train" in record["stages"] and "train" not in started else "evaluate"
                )
                artifact_id = data.get("artifact_id")
                if not artifact_id:
                    issue("Start state publication has no artifact ID", stage=stage)
                else:
                    artifact_reference("start_state_saved", artifact_id, owner=run_id)
                    saved.add(stage)
            elif data.get("status") == "running":
                stage = data["stage"]
                started.add(stage)
                # A saved-measurement reaggregation has no model or prediction input.
                requires_copy = ("train" in record["stages"] or record.get("initial_checkpoint")
                                 or record.get("input_predictions_artifact"))
                if requires_copy and stage not in saved:
                    issue("Stage started before publication of its start state", stage=stage)
        if record.get("training_started") != ("train" in started):
            issue("Training start flag disagrees with history", item="training_started")
        if (record["status"] == "succeeded" and record["stages"].get("train") == "succeeded"
                and "train" not in started):
            issue("Successful training has no start record", stage="train")

    def verify(self, run=None):
        with self.transaction() as db:
            if run is not None and not db.execute("SELECT 1 FROM runs WHERE id=?", (run,)).fetchone():
                raise ExperimentError("not_found", "Run not found", run_id=run)
            records = (db.execute("SELECT * FROM runs ORDER BY rowid") if run is None else
                       db.execute("SELECT * FROM runs WHERE id=?", (run,)))
            report = self.artifacts.verify(db, run)
            for row in records:
                report["checked"] += 1
                report["issues"].extend(verify_commits(db, row["id"]))
                try:
                    record = json.loads(row["record"])
                    for key in ("id", "branch", "sequence", "status"):
                        if record[key] != row[key]:
                            raise ValueError(f"{key} disagrees with the indexed run")
                    if not isinstance(record["stages"], dict):
                        raise ValueError("Invalid stages")
                    self._verify_run(db, record, report)
                except (TypeError, ValueError, KeyError, sqlite3.ProgrammingError) as exc:
                    report["issues"].append({"item": "run_record", "run_id": row["id"],
                                             "reason": f"Invalid run record: {exc}"})
            if run is None:
                # Detect broken references even in tables without a public ID
                # (for example an aggregate whose input record was removed).
                for table, rowid, parent, _ in db.execute("PRAGMA foreign_key_check"):
                    report["issues"].append({"table": table, "row": rowid,
                                             "reason": f"Missing reference to {parent}"})
            else:
                report["ready"] = not report["issues"]
            return report
