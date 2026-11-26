"""Evaluation journal, direct provenance and snapshot reads.

Immutable values live in the run's events. These secondary indexes make them
addressable without loading the full history. The global event sequence also
provides an as-of boundary for a cursor, including evaluation state changes.
"""

from __future__ import annotations

import base64
import binascii
import json
from uuid import uuid4

from .errors import ExperimentError


EVALUATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    run_id TEXT NOT NULL REFERENCES runs(id),
    evaluation_id TEXT NOT NULL,
    request_id TEXT,
    started INTEGER NOT NULL UNIQUE REFERENCES events(id),
    finished INTEGER REFERENCES events(id),
    PRIMARY KEY (run_id, evaluation_id),
    UNIQUE (run_id, request_id)
);
CREATE INDEX IF NOT EXISTS eval_by_run ON evaluations(run_id, started);
CREATE INDEX IF NOT EXISTS eval_logs_by_run ON events(run_id, id) WHERE kind='evaluation_log';
CREATE TABLE IF NOT EXISTS evaluation_sources (
    source_run TEXT NOT NULL,
    source_eval TEXT NOT NULL,
    derived_event INTEGER NOT NULL REFERENCES evaluations(started),
    relation TEXT NOT NULL,
    PRIMARY KEY (source_run, source_eval, derived_event),
    FOREIGN KEY (source_run, source_eval) REFERENCES evaluations(run_id, evaluation_id)
);
CREATE TABLE IF NOT EXISTS evaluation_objects (
    event_id INTEGER PRIMARY KEY REFERENCES events(id),
    run_id TEXT NOT NULL,
    evaluation_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    UNIQUE (run_id, evaluation_id, kind, object_id),
    FOREIGN KEY (run_id, evaluation_id) REFERENCES evaluations(run_id, evaluation_id)
);
CREATE INDEX IF NOT EXISTS eval_object_sequence
    ON evaluation_objects(run_id, evaluation_id, kind, event_id);
CREATE TABLE IF NOT EXISTS aggregate_inputs (
    aggregate_event INTEGER NOT NULL REFERENCES evaluation_objects(event_id),
    position INTEGER NOT NULL,
    input_event INTEGER NOT NULL REFERENCES evaluation_objects(event_id),
    reference TEXT NOT NULL,
    PRIMARY KEY (aggregate_event, position)
);
CREATE INDEX IF NOT EXISTS eval_used_by ON aggregate_inputs(input_event, aggregate_event);
CREATE TABLE IF NOT EXISTS decision_inputs (
    decision_event INTEGER PRIMARY KEY REFERENCES events(id),
    aggregate_event INTEGER NOT NULL REFERENCES evaluation_objects(event_id)
);
CREATE INDEX IF NOT EXISTS eval_decisions_by_input ON decision_inputs(aggregate_event);
"""

_MAX_EVENT = 2**63 - 1
_MAX_PAGE = 1000
_END_STATES = ("succeeded", "failed", "interrupted")


def _invalid(message, **details):
    return ExperimentError("invalid_request", message, **details)


def _missing(kind, **details):
    return ExperimentError("not_found", f"Unknown {kind}", kind=kind, **details)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise _invalid("Expected finite, JSON-serializable evaluation data", reason=str(exc)) from exc


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise _invalid(f"{name} must be a non-empty string", field=name)
    return value


def _compare(saved, proposed, **ids):
    """Compare the serialized form so retries preserve types as well as values."""
    if _json(saved) != _json(proposed):
        keys = set(saved) | set(proposed)
        fields = sorted(k for k in keys if k not in saved or k not in proposed
                        or _json(saved[k]) != _json(proposed[k]))
        raise _invalid("Conflicting evaluation record", fields=fields, **ids)


def _evaluation(db, run, evaluation):
    row = db.execute(
        "SELECT started, finished FROM evaluations WHERE run_id=? AND evaluation_id=?",
        (run, evaluation),
    ).fetchone()
    if row is None:
        raise _missing("evaluation", run_id=run, evaluation_id=evaluation)
    return row


def _open(row, run, evaluation):
    if row["finished"] is not None:
        raise _invalid("Evaluation is already closed", run_id=run, evaluation_id=evaluation)


def _payload(db, event_id):
    return json.loads(db.execute("SELECT payload FROM events WHERE id=?", (event_id,)).fetchone()[0])


def _object(db, run, evaluation, kind, identifier, *, required=True):
    row = db.execute(
        "SELECT event_id FROM evaluation_objects "
        "WHERE run_id=? AND evaluation_id=? AND kind=? AND object_id=?",
        (run, evaluation, kind, identifier),
    ).fetchone()
    if row is None and required:
        raise _missing(kind, run_id=run, evaluation_id=evaluation, **{f"{kind}_id": identifier})
    return row[0] if row else None


def _page(db, scope, limit, cursor, sql, params, key, unpack, *, initial_position=0):
    if type(limit) is not int or not 1 <= limit <= _MAX_PAGE:
        raise _invalid("limit must be a positive integer within the page limit", max_limit=_MAX_PAGE)
    if cursor is None:
        until = db.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
        position = initial_position
    else:
        try:
            version, saved_scope, until, position = json.loads(
                base64.b64decode(cursor, altchars=b"-_", validate=True)
            )
            if (version != 1 or saved_scope != scope or type(until) is not int
                    or type(position) is not int or not 0 <= until <= _MAX_EVENT
                    or not 0 <= position <= _MAX_EVENT):
                raise ValueError
        except (ValueError, TypeError, binascii.Error) as exc:
            raise _invalid("Invalid cursor or cursor belongs to a different query") from exc
    rows = db.execute(sql, {**params, "until": until, "pos": position, "limit": limit + 1}).fetchall()
    next_cursor = None
    if len(rows) > limit:
        rows.pop()
        token = _json([1, scope, until, rows[-1]["pos"]]).encode()
        next_cursor = base64.urlsafe_b64encode(token).decode()
    return {key: [unpack(row) for row in rows], "next_cursor": next_cursor}, until


_EVAL_SELECT = """
    SELECT v.started AS pos, s.payload AS started, f.payload AS finished, m.payload AS measured_model,
        (SELECT json_group_object(json_extract(l.payload, '$.component'), json_extract(l.payload, '$.version'))
         FROM events l WHERE l.run_id=v.run_id AND l.kind='evaluation_log'
           AND l.id BETWEEN v.started AND min(coalesce(v.finished, :until), :until)
           AND json_extract(l.payload, '$.evaluation_id')=v.evaluation_id
           AND json_extract(l.payload, '$.message')='dataset_opened'
           AND json_type(l.payload, '$.component')='text'
           AND json_type(l.payload, '$.version')='text') AS opened_versions
    FROM evaluations v
    JOIN events s ON s.id=v.started
    LEFT JOIN events f ON f.id=v.finished AND f.id<=:until
    LEFT JOIN evaluation_objects o ON o.run_id=v.run_id AND o.evaluation_id=v.evaluation_id
         AND o.kind='target' AND o.object_id='model'
    LEFT JOIN events m ON m.id=o.event_id AND m.id<=:until
"""


def _describe(row):
    start = json.loads(row["started"])
    end = json.loads(row["finished"]) if row["finished"] is not None else {}
    if row["measured_model"] is not None:
        start["target"]["model_sha256"] = json.loads(row["measured_model"])["model_sha256"]
    # The request contains the versions pinned at acceptance. An evaluator can
    # also open a dataset whose version is first assigned during this run.
    conditions = start["conditions"]
    if isinstance(conditions, dict) and isinstance(conditions.get("versions"), dict):
        conditions["versions"].update(json.loads(row["opened_versions"]))
    return {
        "run_id": start["run_id"], "evaluation_id": start["evaluation_id"],
        "target": start["target"], "conditions": start["conditions"],
        "relation": start["relation"], "source_evaluations": start["source_evaluations"],
        "state": "running", "complete": None, "reason": None, **end,
    }


def _event_data(row):
    return json.loads(row["data"])


def _source_refs(sources, relation, reason):
    if relation is not None:
        _identifier(relation, "relation")
    refs = []
    seen = set()
    for source in sources:
        if isinstance(source, dict):
            run, evaluation = source.get("run_id"), source.get("evaluation_id")
        elif isinstance(source, (tuple, list)) and len(source) == 2:
            run, evaluation = source
        else:
            raise _invalid("Expected a source run and evaluation")
        ref = {"run_id": _identifier(run, "run_id"),
               "evaluation_id": _identifier(evaluation, "evaluation_id"), "relation": relation}
        if reason is not None:
            ref["reason"] = reason
        key = (ref["run_id"], ref["evaluation_id"])
        if key in seen:
            raise _invalid("Duplicate source evaluation", **ref)
        seen.add(key)
        refs.append(ref)
    if bool(refs) != bool(relation):
        raise _invalid("A derived evaluation needs both a relation and source evaluations")
    if relation == "correction" and not reason:
        raise _invalid("A correction needs a reason", relation=relation)
    return refs


class EvaluationHistory:
    """Read and write evaluation records in a project's Store."""

    def __init__(self, store):
        self.store = store

    def begin_evaluation(self, run_id, target, conditions, *, request_id=None,
                         relation=None, source_evaluations=(), reason=None, on_create=None):
        """Reserve an evaluation and acquire any input lease in the same transaction.

        Only the creator prepares the inputs and runs the measurement. A retry
        can refer to the original evaluation even after its lease is released.
        """
        if request_id is not None:
            _identifier(request_id, "request_id")
        sources = _source_refs(source_evaluations, relation, reason)
        request = {
            "run_id": run_id, "target": target, "conditions": conditions,
            "relation": relation, "source_evaluations": sources, "reason": reason,
            "request_id": request_id,
        }
        _json(request)
        with self.store.transaction(write=True) as db:
            run = self.store.run(db, run_id)
            if request_id is not None:
                row = db.execute("SELECT evaluation_id, started FROM evaluations "
                                 "WHERE run_id=? AND request_id=?", (run_id, request_id)).fetchone()
                if row:
                    original = _payload(db, row["started"])
                    original.pop("evaluation_id")
                    _compare(original, request, request_id=request_id, run_id=run_id)
                    return row["evaluation_id"], False
            if run["status"] in _END_STATES:
                raise _invalid("A new evaluation needs an active run", run_id=run_id,
                               status=run["status"])
            for source in sources:
                _evaluation(db, source["run_id"], source["evaluation_id"])
            eid = str(uuid4())
            event_id = self.store.event(db, run_id, "evaluation_started",
                                        {**request, "evaluation_id": eid})
            db.execute("INSERT INTO evaluations(run_id,evaluation_id,request_id,started) "
                       "VALUES(?,?,?,?)", (run_id, eid, request_id, event_id))
            db.executemany("INSERT INTO evaluation_sources VALUES(?,?,?,?)",
                           [(s["run_id"], s["evaluation_id"], event_id, relation) for s in sources])
            if on_create is not None:
                on_create(db, eid)
            return eid, True

    def log_evaluation(self, run_id, evaluation_id, message, **details):
        data = {**details, "run_id": run_id, "evaluation_id": evaluation_id, "message": message}
        _json(data)
        with self.store.transaction(write=True) as db:
            _evaluation(db, run_id, evaluation_id)
            self.store.event(db, run_id, "evaluation_log", data)

    def identify_model(self, run_id, evaluation_id, digest):
        """Record the frozen model after registration and before the first measurement."""
        data = {"run_id": run_id, "evaluation_id": evaluation_id, "model_sha256": digest}
        with self.store.transaction(write=True) as db:
            row = _evaluation(db, run_id, evaluation_id)
            previous = _object(db, run_id, evaluation_id, "target", "model", required=False)
            if previous:
                _compare(_payload(db, previous), data, evaluation_id=evaluation_id)
                return
            _open(row, run_id, evaluation_id)
            event = self.store.event(db, run_id, "evaluation_model_frozen", data)
            db.execute("INSERT INTO evaluation_objects VALUES(?,?,?,?,?)",
                       (event, run_id, evaluation_id, "target", "model"))

    def record_result(self, run_id, evaluation_id, result_id, target, value):
        _identifier(result_id, "result_id")
        data = {"run_id": run_id, "evaluation_id": evaluation_id,
                "result_id": result_id, "target": target, "value": value}
        _json(data)
        with self.store.transaction(write=True) as db:
            row = _evaluation(db, run_id, evaluation_id)
            previous = _object(db, run_id, evaluation_id, "result", result_id, required=False)
            if previous is not None:
                saved = _payload(db, previous)
                _compare(saved, data, evaluation_id=evaluation_id, result_id=result_id)
                return saved
            _open(row, run_id, evaluation_id)
            event_id = self.store.event(db, run_id, "evaluation_result", data)
            db.execute("INSERT INTO evaluation_objects VALUES(?,?,?,?,?)",
                       (event_id, run_id, evaluation_id, "result", result_id))
            return data

    def publish_aggregate(self, run_id, evaluation_id, aggregate_id, value, conditions,
                          provisional, inputs):
        _identifier(aggregate_id, "aggregate_id")
        if type(provisional) is not bool:
            raise _invalid("provisional must be a boolean", field="provisional")
        data = {"run_id": run_id, "evaluation_id": evaluation_id, "aggregate_id": aggregate_id,
                "value": value, "conditions": conditions, "provisional": provisional}
        _json(data)
        references = list(inputs)
        with self.store.transaction(write=True) as db:
            row = _evaluation(db, run_id, evaluation_id)
            existing = _object(db, run_id, evaluation_id, "aggregate", aggregate_id, required=False)
            if existing is not None:
                saved = _payload(db, existing)
                old_inputs = [json.loads(r[0]) for r in db.execute(
                    "SELECT reference FROM aggregate_inputs WHERE aggregate_event=? ORDER BY position",
                    (existing,),
                )]
                _compare({**saved, "inputs": old_inputs}, {**data, "inputs": references},
                         evaluation_id=evaluation_id, aggregate_id=aggregate_id)
                return saved
            _open(row, run_id, evaluation_id)
            rows = []
            for index, ref in enumerate(references, 1):
                if not isinstance(ref, dict) or ref.get("kind") not in ("result", "aggregate"):
                    raise _invalid("Expected a result or aggregate input reference", position=index)
                kind = ref["kind"]
                key = f"{kind}_id"
                source = _object(db, _identifier(ref.get("run_id"), "run_id"),
                                 _identifier(ref.get("evaluation_id"), "evaluation_id"),
                                 kind, _identifier(ref.get(key), key))
                rows.append((index, source, _json(ref)))
            event_id = self.store.event(db, run_id, "aggregate_published", data)
            db.execute("INSERT INTO evaluation_objects VALUES(?,?,?,?,?)",
                       (event_id, run_id, evaluation_id, "aggregate", aggregate_id))
            db.executemany("INSERT INTO aggregate_inputs VALUES(?,?,?,?)",
                           [(event_id, *item) for item in rows])
            return data

    def finish_evaluation(self, run_id, evaluation_id, state, *, complete=None,
                          reason=None, aggregate_ids=(), **details):
        if state not in _END_STATES:
            raise _invalid("Expected a terminal evaluation state", state=state)
        if complete is not None and type(complete) is not bool:
            raise _invalid("complete must be a boolean or null", field="complete")
        if state == "succeeded" and complete is not True:
            raise _invalid("Successful evaluations require a positive completion report")
        aggregates = [_identifier(a, "aggregate_id") for a in aggregate_ids]
        data = {**details, "run_id": run_id, "evaluation_id": evaluation_id,
                "state": state, "complete": complete, "reason": reason, "aggregate_ids": aggregates}
        _json(data)
        with self.store.transaction(write=True) as db:
            row = _evaluation(db, run_id, evaluation_id)
            if row["finished"] is not None:
                saved = _payload(db, row["finished"])
                _compare(saved, data, evaluation_id=evaluation_id)
                return saved
            missing = [a for a in aggregates
                       if _object(db, run_id, evaluation_id, "aggregate", a, required=False) is None]
            if missing:
                raise _invalid("Required aggregates have not been published",
                               evaluation_id=evaluation_id, missing_aggregate_ids=missing)
            _close(self.store, db, run_id, evaluation_id, data)
            return data

    def record_decision(self, run, evaluation, decision, *, aggregate_id=None,
                        followup_run=None, request_id=None):
        data = {"run_id": run, "evaluation_id": evaluation, "decision": decision,
                "aggregate_id": aggregate_id, "followup_run": followup_run}
        _json(data)
        with self.store.transaction(write=True) as db:
            previous = self.store.request(db, request_id, "record_decision", data)
            if previous is not None:
                return previous
            _evaluation(db, run, evaluation)
            if followup_run is not None:
                self.store.run(db, followup_run)
            source = None
            if aggregate_id is not None:
                source = _object(db, run, evaluation, "aggregate", aggregate_id)
            decision_id = str(uuid4())
            event_id = self.store.event(db, run, "decision_recorded", {**data, "decision_id": decision_id})
            if source is not None:
                db.execute("INSERT INTO decision_inputs VALUES(?,?)", (event_id, source))
            self.store.receipt(db, request_id, "record_decision", data, decision_id)
            return decision_id

    def get_evaluation(self, run_id, evaluation_id):
        with self.store.transaction() as db:
            params = {"run": run_id, "eid": evaluation_id, "until": _MAX_EVENT}
            row = db.execute(_EVAL_SELECT + " WHERE v.run_id=:run AND v.evaluation_id=:eid",
                             params).fetchone()
            if row is None:
                raise _missing("evaluation", run_id=run_id, evaluation_id=evaluation_id)
            count = db.execute("SELECT count(*) FROM evaluation_objects WHERE run_id=:run "
                               "AND evaluation_id=:eid AND kind='result'", params).fetchone()[0]
            return {**_describe(row), "result_count": count}

    def list_evaluations(self, run_id, *, limit, cursor=None):
        with self.store.transaction() as db:
            self.store.run(db, run_id)
            sql = _EVAL_SELECT + """ WHERE v.run_id=:run AND v.started>:pos AND v.started<=:until
                                     ORDER BY v.started LIMIT :limit"""
            return _page(db, ["evaluations", run_id], limit, cursor, sql, {"run": run_id},
                         "evaluations", _describe)[0]

    def list_derived_evaluations(self, run_id, evaluation_id, *, relation=None, limit, cursor=None):
        with self.store.transaction() as db:
            _evaluation(db, run_id, evaluation_id)
            sql = _EVAL_SELECT + """
                JOIN evaluation_sources d ON d.derived_event=v.started
                WHERE d.source_run=:run AND d.source_eval=:eid
                  AND (:relation IS NULL OR d.relation=:relation)
                  AND v.started>:pos AND v.started<=:until ORDER BY v.started LIMIT :limit
            """
            params = {"run": run_id, "eid": evaluation_id, "relation": relation}
            return _page(db, ["derived", run_id, evaluation_id, relation], limit, cursor, sql,
                         params, "evaluations", _describe)[0]

    def read_results(self, run_id, evaluation_id, *, limit, cursor=None, after=None):
        if cursor is not None and after is not None:
            raise _invalid("Use only one of cursor and after")
        with self.store.transaction() as db:
            evaluation = _evaluation(db, run_id, evaluation_id)
            position = _object(db, run_id, evaluation_id, "result", after) if after is not None else 0
            page, until = self._objects(db, run_id, evaluation_id, "result", limit, cursor, position)
            finished = evaluation["finished"]
            page["evaluation_state"] = (_payload(db, finished)["state"]
                                        if finished is not None and finished <= until else "running")
            return page

    def list_aggregates(self, run_id, evaluation_id, *, limit, cursor=None):
        with self.store.transaction() as db:
            _evaluation(db, run_id, evaluation_id)
            return self._objects(db, run_id, evaluation_id, "aggregate", limit, cursor, 0)[0]

    @staticmethod
    def _objects(db, run, evaluation, kind, limit, cursor, position):
        sql = """SELECT o.event_id AS pos, e.payload AS data
                 FROM evaluation_objects o JOIN events e ON e.id=o.event_id
                 WHERE o.run_id=:run AND o.evaluation_id=:eid AND o.kind=:kind
                   AND o.event_id>:pos AND o.event_id<=:until ORDER BY o.event_id LIMIT :limit"""
        params = {"run": run, "eid": evaluation, "kind": kind}
        return _page(db, [kind, run, evaluation], limit, cursor, sql, params, kind + "s",
                     _event_data, initial_position=position)

    def get_result(self, run_id, evaluation_id, result_id):
        with self.store.transaction() as db:
            return _payload(db, _object(db, run_id, evaluation_id, "result", result_id))

    def get_aggregate(self, run_id, evaluation_id, aggregate_id):
        with self.store.transaction() as db:
            return _payload(db, _object(db, run_id, evaluation_id, "aggregate", aggregate_id))

    def read_aggregate_inputs(self, run_id, evaluation_id, aggregate_id, *, limit, cursor=None):
        with self.store.transaction() as db:
            event_id = _object(db, run_id, evaluation_id, "aggregate", aggregate_id)
            sql = """SELECT position AS pos, reference AS data FROM aggregate_inputs
                     WHERE aggregate_event=:event AND position>:pos ORDER BY position LIMIT :limit"""
            return _page(db, ["inputs", run_id, evaluation_id, aggregate_id], limit, cursor,
                         sql, {"event": event_id}, "inputs", _event_data)[0]

    def list_uses(self, run_id, evaluation_id, *, result_id=None, aggregate_id=None, limit,
                  cursor=None):
        if (result_id is None) == (aggregate_id is None):
            raise _invalid("Specify exactly one of result_id and aggregate_id")
        kind, identifier = ("result", result_id) if result_id is not None else ("aggregate", aggregate_id)
        with self.store.transaction() as db:
            event_id = _object(db, run_id, evaluation_id, kind, identifier)
            sql = """
                SELECT e.id AS pos, e.kind, e.payload AS data FROM events e
                WHERE e.id IN (
                    SELECT aggregate_event FROM aggregate_inputs WHERE input_event=:event
                    UNION SELECT decision_event FROM decision_inputs WHERE aggregate_event=:event
                ) AND e.id>:pos AND e.id<=:until ORDER BY e.id LIMIT :limit
            """
            def describe_use(row):
                data = _event_data(row)
                use = "decision" if row["kind"] == "decision_recorded" else "aggregate"
                name = use + "_id"
                return {"kind": use, "run_id": data["run_id"], "evaluation_id": data["evaluation_id"],
                        name: data[name]}

            return _page(db, ["uses", run_id, evaluation_id, kind, identifier], limit, cursor,
                         sql, {"event": event_id}, "uses", describe_use)[0]


def _close(store, db, run, evaluation, data):
    event_id = store.event(db, run, "evaluation_finished", data)
    db.execute("UPDATE evaluations SET finished=? WHERE run_id=? AND evaluation_id=?",
               (event_id, run, evaluation))


def fail_open_evaluations(store, db, run_id, reason, *, state="failed"):
    """Finalize orphaned evaluations in the same transaction that releases a run."""
    rows = db.execute("SELECT evaluation_id FROM evaluations WHERE run_id=? AND finished IS NULL",
                      (run_id,)).fetchall()
    for row in rows:
        evaluation_id = row["evaluation_id"]
        data = {"run_id": run_id, "evaluation_id": evaluation_id, "state": state,
                "complete": None, "reason": reason, "aggregate_ids": []}
        _close(store, db, run_id, evaluation_id, data)


def evaluation_summary(db, run_id):
    """Most recent evaluation for each configured evaluator, without per-sample records."""
    rows = db.execute(_EVAL_SELECT + " WHERE v.run_id=:run ORDER BY v.started",
                      {"run": run_id, "until": _MAX_EVENT})
    latest = {}
    for row in rows:
        data = _describe(row)
        target = data["target"] if isinstance(data["target"], dict) else {}
        conditions = data["conditions"] if isinstance(data["conditions"], dict) else {}
        component = conditions.get("component", target.get("component", "evaluator"))
        latest[component] = (data, target, conditions)
    summary = []
    for component, (data, target, conditions) in latest.items():
        eid = data["evaluation_id"]
        aggregates = db.execute(
            "SELECT e.payload FROM evaluation_objects o JOIN events e ON e.id=o.event_id "
            "WHERE o.run_id=? AND o.evaluation_id=? AND o.kind='aggregate' ORDER BY o.event_id",
            (run_id, eid),
        )
        item = {
            "component": component, "evaluation_id": eid, "state": data["state"],
            "complete": data["complete"], "checkpoint_id": target.get("checkpoint_id"),
            "progress": target.get("progress"), "split": conditions.get("split", target.get("split")),
            "aggregates": [],
        }
        if target.get("metric") is not None:
            item["metric"] = target["metric"]
        for source in (conditions, target, data.get("result") or {}):
            if isinstance(source, dict) and "selection" in source:
                item["selection"] = source["selection"]
        for row in aggregates:
            aggregate = json.loads(row[0])
            item["aggregates"].append({k: aggregate[k] for k in ("aggregate_id", "value", "provisional")})
        summary.append(item)
    return summary
