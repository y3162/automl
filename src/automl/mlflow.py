"""Project the local experiment history into MLflow Tracking."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from automl.paths import work_directory


_TERMINAL = {"succeeded": "FINISHED", "failed": "FAILED", "interrupted": "KILLED"}


def _timestamp(date):
    return int(datetime.fromisoformat(date).timestamp() * 1000)


def _key(name):
    # MLflow restricts tracking keys. Keep disallowed names distinct after conversion.
    if len(name) <= 240 and re.fullmatch(r"[\w. -]+", name):
        return name
    readable = re.sub(r"[^\w. -]+", "_", name)[:200]
    return readable + "." + hashlib.sha256(name.encode()).hexdigest()[:16]


def _step(progress, event_id):
    if type(progress) is int:
        return progress
    if not isinstance(progress, dict):
        return event_id
    for key in ("step", "rounds", "epoch"):
        value = progress.get(key)
        if type(value) is int:
            return value
    return event_id


def _metrics(events, after):
    targets = {}
    for event in events:
        kind, data, event_id = event["kind"], event["payload"], event["id"]
        if kind == "evaluation_started":
            target = data["target"]
            targets[data["evaluation_id"]] = target if isinstance(target, dict) else {}
        if event_id <= after:
            continue
        if kind == "training_progress":
            step = _step(data.get("progress", data), event_id)
            values = {f"train.{k}": v for k, v in data.get("metrics", data).items()
                      if k not in {"rounds", "step", "epoch"}}
        elif kind == "aggregate_published":
            target = targets[data["evaluation_id"]]
            step = _step(target.get("progress"), event_id)
            conditions = data["conditions"]
            split = conditions.get("split") if isinstance(conditions, dict) else None
            split = split or target.get("component", "evaluation")
            prefix = "provisional" if data["provisional"] else "eval"
            values = {f"{prefix}.{split}.{data['aggregate_id']}": data["value"]}
        else:
            continue
        for key, value in values.items():
            if type(value) in (int, float) and math.isfinite(value):
                yield _key(key), float(value), _timestamp(event["created_at"]), step


def _publish(client, mlrun, history, confirmed):
    run_id = mlrun.info.run_id
    tags = mlrun.data.tags
    revision = str(history["revision"])
    record = history["runs"][0]
    confirmed = str(confirmed or record["status"] in _TERMINAL).lower()
    if tags.get("automl.state_confirmed") != confirmed:
        client.set_tag(run_id, "automl.state_confirmed", confirmed)
    if tags.get("automl.revision") == revision and tags.get("automl.sync_state") == "synced":
        return

    recovering = tags.get("automl.sync_state") in {"syncing", "error"}
    client.set_tag(run_id, "automl.sync_state", "syncing")
    try:
        for name, spec in record["config"]["components"].items():
            params = {f"settings.{k}": v for k, v in spec.get("settings", {}).items()
                      if isinstance(v, (str, int, float, bool)) and len(str(v)) <= 500}
            for key, value in {"implementation": spec["implementation"], **params}.items():
                client.log_param(run_id, _key(f"{name}.{key}"), value)
        # Stable event timestamps and steps allow recovery after a lost log_metric response.
        previous = {}
        for key, value, time, step in _metrics(history["events"], int(tags.get("automl.last_event", 0))):
            if recovering:
                if key not in previous:
                    previous[key] = {(m.value, m.timestamp, m.step) for m in client.get_metric_history(run_id, key)}
                if (value, time, step) in previous[key]:
                    continue
            client.log_metric(run_id, key, value, timestamp=time, step=step)
        client.set_tag(run_id, "automl.branch", record["branch"])
        client.set_tag(run_id, "automl.status", record["status"])
        client.log_dict(run_id, record["config"], "automl/config.json")
        # Preserve checkpoint choices, dataset versions and later decisions in this
        # snapshot; per-target measurements remain queryable by AutoML run ID.
        selected = {"evaluation_started", "decision_recorded", "dataset_opened", "dataset_split"}
        lineage = {
            "stages": record["stages"], "initial_checkpoint": record.get("initial_checkpoint"),
            "events": [event for event in history["events"] if event["kind"] in selected],
            "artifacts": history["artifacts"],
        }
        client.log_dict(run_id, lineage, "automl/lineage.json")
        if record["status"] in _TERMINAL:
            client.set_terminated(run_id, _TERMINAL[record["status"]], end_time=_timestamp(record["ended_at"]))
        last_event = history["events"][-1]["id"] if history["events"] else 0
        client.set_tag(run_id, "automl.last_event", str(last_event))
        client.set_tag(run_id, "automl.revision", revision)
        client.set_tag(run_id, "automl.sync_state", "synced")
    except Exception:
        client.set_tag(run_id, "automl.sync_state", "error")
        raise


def _tracking_directory(root: Path) -> Path:
    return work_directory(root, fallback=root.parent / "work") / "mlflow"


def sync(experiment, client, *, experiment_name="AutoML", artifact_location=None):
    """Sync one project's revisions and return the AutoML-to-MLflow run-ID mapping.

    `client` is an MLflow Tracking MlflowClient. A tracking experiment is bound to
    the source store's portable ID, which is preserved when the store is copied.
    """
    store = experiment.store
    project_id = store.project_id
    work = _tracking_directory(store.root)
    work.mkdir(parents=True, exist_ok=True)
    with (work / f"{project_id}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        target = client.get_experiment_by_name(experiment_name)
        if target is None:
            eid = client.create_experiment(experiment_name, artifact_location=artifact_location,
                                           tags={"automl.project_id": project_id})
        else:
            eid = target.experiment_id
            if target.tags.get("automl.project_id") != project_id:
                raise ValueError(f"MLflow experiment {experiment_name!r} is not bound to this AutoML project")
        result = {}
        for run_id, revision in experiment.tracking_revisions().items():
            state = experiment.status(run_id, reconcile=True)
            matches = client.search_runs([eid], f"tags.`automl.run_id` = '{run_id}'", max_results=2)
            if len(matches) > 1:
                raise ValueError(f"multiple MLflow runs reference {run_id}")
            if matches and state["status"] in _TERMINAL:
                tags = matches[0].data.tags
                if (tags.get("automl.revision") == str(revision) and tags.get("automl.sync_state") == "synced"
                        and tags.get("automl.status") == state["status"]):
                    result[run_id] = matches[0].info.run_id
                    continue
            history = experiment.tracking_snapshot(run_id)
            mlrun = matches[0] if matches else client.create_run(
                eid, start_time=_timestamp(history["runs"][0]["created_at"]),
                tags={"automl.run_id": run_id, "automl.project_id": project_id},
            )
            _publish(client, mlrun, history, state["state_confirmed"])
            result[run_id] = mlrun.info.run_id
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/store"))
    parser.add_argument("--tracking-uri", help="default: SQLite under data/work/mlflow")
    parser.add_argument("--experiment", default="AutoML")
    args = parser.parse_args(argv)
    if not (args.root / "history.sqlite3").is_file():
        parser.error(f"AutoML store does not exist: {args.root}")
    try:
        from mlflow import MlflowClient
    except ImportError:
        parser.error("install the mlflow optional dependency to export tracking data")
    from automl.experiment import Experiment

    work = _tracking_directory(args.root.resolve())
    work.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=args.tracking_uri or f"sqlite:///{work}/tracking.db")
    artifacts = (work / "artifacts").as_uri() if args.tracking_uri is None else None
    result = sync(Experiment(args.root), client, experiment_name=args.experiment, artifact_location=artifacts)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
