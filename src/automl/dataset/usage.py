"""Record actual input use and bind durable, per-process prefixes to checkpoints.

Each add covers one ordered sequence of inputs from one process. Positions start
at one *in this run*, including after a resume. A distributed batch has one
submission per declared process. Callers supply a store transaction so a
checkpoint and its validated usage range become visible together.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence


USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_usage_batches (
    run_id TEXT NOT NULL REFERENCES runs(id),
    batch_id TEXT NOT NULL,
    processes TEXT NOT NULL,
    PRIMARY KEY (run_id, batch_id)
);
CREATE TABLE IF NOT EXISTS dataset_usage (
    run_id TEXT NOT NULL REFERENCES runs(id),
    usage_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK(position >= 1),
    process INTEGER NOT NULL CHECK(process >= 0),
    dataset_version TEXT NOT NULL,
    split TEXT,
    sample_ids TEXT NOT NULL,
    batch_id TEXT,
    details TEXT NOT NULL,
    PRIMARY KEY (run_id, usage_id),
    UNIQUE (run_id, process, position),
    UNIQUE (run_id, batch_id, process),
    FOREIGN KEY (run_id, batch_id) REFERENCES dataset_usage_batches(run_id, batch_id)
);
CREATE TABLE IF NOT EXISTS dataset_usage_prefixes (
    run_id TEXT NOT NULL REFERENCES runs(id),
    process INTEGER NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (run_id, process)
);
CREATE TABLE IF NOT EXISTS dataset_usage_commits (
    checkpoint_id TEXT PRIMARY KEY REFERENCES artifacts(id),
    run_id TEXT NOT NULL REFERENCES runs(id),
    usage_range TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS dataset_usage_commits_by_run ON dataset_usage_commits(run_id);
"""


def _invalid(message, **details):
    # Dataset contracts can be imported before the Experiment facade is loaded.
    from automl.experiment.errors import ExperimentError

    raise ExperimentError("invalid_request", message, **details)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        _invalid(f"usage data must be JSON serializable: {exc}")


def _number(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        _invalid(f"{name} must be an integer >= {minimum}", field=name)
    return value


def _id(value, name):
    if not isinstance(value, str) or not value:
        _invalid(f"{name} must be a non-empty string", field=name)
    return value


def _batch_key(batch_id):
    if isinstance(batch_id, str):
        _id(batch_id, "batch_id")
    else:
        _number(batch_id, "batch_id")
    return _json(batch_id)


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _invalid(f"{name} must be a sequence", field=name)
    return value


def _members(processes, process):
    if type(processes) is int:
        count = _number(processes, "processes", minimum=1)
        if process >= count:
            _invalid("process is outside the declared batch", process=process)
        return list(range(count))
    members = sorted(_number(p, "process") for p in _sequence(processes, "processes"))
    if process not in members or len(set(members)) != len(members):
        _invalid("processes must be unique and include the submitting process", process=process)
    return members


def _advance_prefix(db, run_id, process, position):
    record = db.execute(
        "SELECT position FROM dataset_usage_prefixes WHERE run_id=? AND process=?",
        (run_id, process),
    ).fetchone()
    prefix = record[0] if record else 0
    if position != prefix + 1:
        return

    # A delayed record can close a gap before an already-received suffix.
    following = db.execute(
        "SELECT position FROM dataset_usage WHERE run_id=? AND process=? AND position>? ORDER BY position",
        (run_id, process, position),
    )
    for (received,) in following:
        if received != position + 1:
            break
        position = received
    db.execute(
        """INSERT INTO dataset_usage_prefixes VALUES (?, ?, ?)
           ON CONFLICT(run_id, process) DO UPDATE SET position=excluded.position""",
        (run_id, process, position),
    )


def add(db, run_id, *, usage_id, position, sample_ids, dataset_version, split=None,
        process=0, batch_id=None, processes=None, details=None):
    """Record one actual use, preserving order and repeated sample IDs on retry."""
    _id(usage_id, "usage_id")
    _id(dataset_version, "dataset_version")
    _number(position, "position", minimum=1)
    _number(process, "process")
    if split is not None:
        _id(split, "split")
    samples = [_id(i, "sample_id") for i in _sequence(sample_ids, "sample_ids")]
    if details is not None and not isinstance(details, Mapping):
        _invalid("details must be a mapping", field="details")

    key = _batch_key(batch_id) if batch_id is not None else None
    fields = dict(position=position, process=process, dataset_version=dataset_version,
                  split=split, sample_ids=_json(samples), batch_id=key, details=_json(details or {}))
    original = db.execute(
        "SELECT * FROM dataset_usage WHERE run_id=? AND usage_id=?", (run_id, usage_id)
    ).fetchone()
    differences = [k for k, v in fields.items() if original[k] != v] if original else []

    members = previous = None
    if key is not None:
        row = db.execute(
            "SELECT processes FROM dataset_usage_batches WHERE run_id=? AND batch_id=?",
            (run_id, key),
        ).fetchone()
        previous = json.loads(row[0]) if row else None
        declared = processes if processes is not None else (previous or [process])
        members = _members(declared, process)
        if previous is not None and previous != members:
            differences.append("processes")
    elif processes is not None:
        _invalid("processes requires a batch_id", field="processes")
    if differences:
        _invalid("conflicting usage record", usage_id=usage_id, fields=differences)
    if original:
        return

    collision = db.execute(
        """SELECT usage_id FROM dataset_usage WHERE run_id=? AND process=?
           AND (position=? OR batch_id=?)""", (run_id, process, position, key)
    ).fetchone()
    if collision:
        _invalid("this process position or batch already has a usage record",
                 usage_id=usage_id, existing_usage_id=collision[0])
    if key is not None and previous is None:
        db.execute("INSERT INTO dataset_usage_batches VALUES (?,?,?)", (run_id, key, _json(members)))
    db.execute(
        """INSERT INTO dataset_usage
           (run_id,usage_id,position,process,dataset_version,split,sample_ids,batch_id,details)
           VALUES (:run_id, :usage_id, :position, :process, :dataset_version, :split,
                   :sample_ids, :batch_id, :details)""",
        dict(run_id=run_id, usage_id=usage_id, **fields),
    )
    _advance_prefix(db, run_id, process, position)


def _ranges(usage_range):
    if type(usage_range) is int:
        return {0: _number(usage_range, "usage_range")}
    if not isinstance(usage_range, Mapping):
        _invalid("usage_range must give the inclusive position or the positions per process")
    result = {}
    for key, value in usage_range.items():
        rank = int(key) if isinstance(key, str) and key.isascii() and key.isdecimal() else key
        _number(rank, "process")
        if rank in result:
            _invalid("duplicate process in usage_range", process=rank)
        result[rank] = _number(value, "usage_range")
    return result


def missing(db, run_id, usage_range):
    """Return inclusive gaps per process without expanding them to individual IDs."""
    gaps = []
    for process, end in sorted(_ranges(usage_range).items()):
        record = db.execute(
            "SELECT position FROM dataset_usage_prefixes WHERE run_id=? AND process=?", (run_id, process)
        ).fetchone()
        next_position = record[0] + 1 if record else 1
        if next_position > end:
            continue
        rows = db.execute(
            """SELECT position FROM dataset_usage WHERE run_id=? AND process=?
               AND position BETWEEN ? AND ? ORDER BY position""", (run_id, process, next_position, end)
        )
        for (position,) in rows:
            if position > next_position:
                gaps.append(dict(process=process, start=next_position, end=position - 1))
            next_position = position + 1
        if next_position <= end:
            gaps.append(dict(process=process, start=next_position, end=end))
    return gaps


def commit(db, run_id, checkpoint_id, usage_range):
    """Associate a checkpoint with exactly the complete range requested by its caller."""
    ranges = _ranges(usage_range)
    record = _json({str(k): v for k, v in ranges.items()})
    existing = db.execute(
        "SELECT run_id, usage_range FROM dataset_usage_commits WHERE checkpoint_id=?", (checkpoint_id,)
    ).fetchone()
    if existing:
        if tuple(existing) != (run_id, record):
            _invalid("conflicting usage commit", checkpoint_id=checkpoint_id, fields=["usage_range"])
        return
    gaps = missing(db, run_id, ranges)
    if gaps:
        _invalid("usage records are missing", missing=gaps, usage_range=usage_range)
    checkpoint = db.execute(
        "SELECT 1 FROM artifacts WHERE run_id=? AND id=? AND kind='checkpoint'", (run_id, checkpoint_id)
    ).fetchone()
    if checkpoint is None:
        _invalid("checkpoint does not belong to the run", checkpoint_id=checkpoint_id, run_id=run_id)
    db.execute("INSERT INTO dataset_usage_commits VALUES (?,?,?)", (checkpoint_id, run_id, record))


def read_usage(db, run_id, *, batch_id=None, process=None):
    """Read received records and the confirmation status of each declared batch."""
    filters = ["run_id=?"]
    arguments = [run_id]
    if batch_id is not None:
        filters.append("batch_id=?")
        arguments.append(_batch_key(batch_id))
    where_batch = " AND ".join(filters)
    batches = {}
    for key, members in db.execute(
        f"SELECT batch_id,processes FROM dataset_usage_batches WHERE {where_batch} ORDER BY rowid",
        arguments,
    ):
        batches[key] = {"batch_id": json.loads(key), "processes": json.loads(members), "received": set()}

    if process is not None:
        _number(process, "process")
    # Batch completion always refers to all participants, including when the
    # caller requests only one process's inputs.
    for key, rank in db.execute(
        f"SELECT batch_id,process FROM dataset_usage WHERE {where_batch} AND batch_id IS NOT NULL",
        arguments,
    ):
        batches[key]["received"].add(rank)
    for batch in batches.values():
        absent = sorted(set(batch["processes"]) - batch.pop("received"))
        batch.update(missing_processes=absent, complete=not absent)

    if process is not None:
        filters.append("process=?")
        arguments.append(process)
    where = " AND ".join(filters)
    records = []
    for row in db.execute(
        f"SELECT * FROM dataset_usage WHERE {where} ORDER BY position,process", arguments,
    ):
        entry = dict(row)
        for name in ("sample_ids", "batch_id", "details"):
            if entry[name] is not None:
                entry[name] = json.loads(entry[name])
        records.append(entry)
    commits = [
        {"checkpoint_id": checkpoint, "usage_range": json.loads(usage_range)}
        for checkpoint, usage_range in db.execute(
            "SELECT checkpoint_id,usage_range FROM dataset_usage_commits WHERE run_id=? ORDER BY rowid",
            (run_id,),
        )
    ]
    return {"records": records, "batches": list(batches.values()), "commits": commits}


def search(db, run_id, *, condition, minimum=None, maximum=None):
    """Stream individual uses whose recorded transform condition is in range.

    Put numeric conditions in details["reproduction"] when recording use. A
    mapping applies to the whole batch. A list applies one mapping per sample,
    in the order supplied to add. Both bounds are inclusive.
    """
    _id(condition, "condition")
    predicates = ["u.run_id=?", "c.key=?", "c.type IN ('integer','real')"]
    params = [run_id, condition]
    for comparison, bound in ((">=", minimum), ("<=", maximum)):
        if bound is not None:
            if type(bound) not in (float, int) or not math.isfinite(bound):
                _invalid("condition bounds must be finite numbers")
            predicates.append(f"c.value {comparison} ?")
            params.append(float(bound))
    if minimum is not None and maximum is not None and minimum > maximum:
        _invalid("minimum exceeds maximum")
    query = f"""
        SELECT u.usage_id,u.position,u.process,u.dataset_version,u.split,
               s.key AS sample_index,s.value AS sample_id,c.value AS applied_value
        FROM dataset_usage u
        JOIN json_each(u.sample_ids) s
        JOIN json_each(CASE
            WHEN json_type(u.details,'$.reproduction')='object'
                THEN json_extract(u.details,'$.reproduction')
            WHEN json_type(u.details,'$.reproduction[' || s.key || ']')='object'
                THEN json_extract(u.details,'$.reproduction[' || s.key || ']')
        END) c
        WHERE {' AND '.join(predicates)}
        ORDER BY u.position,u.process,s.key
    """
    for row in db.execute(query, params):
        yield dict(row)


def verify_commits(db, run_id):
    """Check saved ranges against the actual ledger, including deleted checkpoints.

    The incremental prefix index drives normal writes. Verify recomputes it from
    the records so a missing use in a copied database is independently detected.
    An unfinished suffix with gaps is valid if no checkpoint claimed that suffix.
    """
    prefixes = dict(db.execute(
        """SELECT process,
                  COALESCE(MIN(CASE WHEN position != expected THEN expected - 1 END),MAX(expected))
             FROM (SELECT process,position,
                          ROW_NUMBER() OVER (PARTITION BY process ORDER BY position) AS expected
                     FROM dataset_usage WHERE run_id=?)
            GROUP BY process""", (run_id,)
    ))
    rows = db.execute(
        """SELECT a.id,a.details,c.run_id AS owner,c.usage_range
             FROM artifacts a LEFT JOIN dataset_usage_commits c ON c.checkpoint_id=a.id
            WHERE a.run_id=? AND a.kind='checkpoint'""", (run_id,)
    )
    from automl.experiment.errors import ExperimentError

    issues = []
    for row in rows:
        try:
            details = json.loads(row["details"])
            if not isinstance(details, dict) or "usage_range" not in details:
                # Older, opaque artifact records can predate the usage ledger.
                if row["usage_range"] is None:
                    continue
                raise ValueError("Missing declared usage range")
            declared = _ranges(details["usage_range"])
            if row["owner"] != run_id or row["usage_range"] is None:
                raise ValueError("Missing matching usage commit")
            committed = _ranges(json.loads(row["usage_range"]))
            if committed != declared:
                raise ValueError("Committed usage range disagrees with the checkpoint")
            for process, end in committed.items():
                if prefixes.get(process, 0) < end:
                    raise ValueError(f"Missing use in process {process} at or before position {end}")
        except (ValueError, TypeError, ExperimentError) as exc:
            issues.append({"run_id": run_id, "artifact_id": row["id"], "item": "usage_range",
                           "reason": str(exc)})
    return issues
