"""Run-scoped operations available to trainers and evaluators."""

from __future__ import annotations

import hashlib
import time
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock

from automl.evaluator import Evaluator, evaluate
from automl.model import Model

from .errors import ExperimentError
from .evaluations import EvaluationHistory
from .state import decode_states, encode_states
from .training import TrainingContext


class Interrupted(Exception):
    """A task accepted a stop request and reached a safe stopping point."""


class RunContext(TrainingContext):
    def __init__(self, store, run_id, *, evaluation_id=None, progress=None):
        self.store = store
        self.run_id = run_id
        self.evaluation_id = evaluation_id
        self.progress = dict(progress or {})
        with store.transaction() as db:
            self.record = store.run(db, run_id)
        self.history = EvaluationHistory(store)
        self.last_evaluation = None
        self._executor = None
        self._lock = Lock()
        self._planned_futures = []

    def stop_requested(self):
        with self.store.transaction() as db:
            return self.store.run(db, self.run_id)["stop_requested"]

    def stage(self, name, state, reason=None):
        with self.store.transaction(write=True) as db:
            self.store.stage(db, self.run_id, name, state, reason=reason)

    def log(self, kind, **payload):
        if self.evaluation_id is not None:
            self.history.log_evaluation(self.run_id, self.evaluation_id, kind, **payload)
        else:
            with self.store.transaction(write=True) as db:
                self.store.event(db, self.run_id, kind, payload)

    def save_artifact(self, name, data):
        """Publish one task output and record which evaluation produced it."""
        if not isinstance(data, bytes):
            raise ValueError("save_artifact requires bytes")
        details = {"name": name, "evaluation_id": self.evaluation_id}
        with self.store.transaction(write=True) as db:
            artifact = self.store.artifacts.save(db, self.run_id, "output", {name: data}, details=details)
            self.store.event(db, self.run_id, "artifact_saved", {**details, "artifact_id": artifact})
        return artifact

    def record_result(self, result_id, target, value):
        return self.history.record_result(self.run_id, self.evaluation_id, result_id, target, value)

    def publish_aggregate(self, aggregate_id, value, conditions, provisional, inputs):
        return self.history.publish_aggregate(
            self.run_id, self.evaluation_id, aggregate_id, value, conditions, provisional, inputs,
        )

    def finish_evaluation(self, result, *, evaluation_id=None):
        """Confirm a task report. Missing aggregate IDs can be published and retried."""
        eid = evaluation_id or self.evaluation_id
        complete = result["complete"]
        failure = None
        if not complete:
            failure = result.get("incomplete_reason") or "Evaluation did not meet its completion condition"
        return self.history.finish_evaluation(
            self.run_id, eid, "succeeded" if complete else "failed", complete=complete,
            reason=failure, aggregate_ids=result.get("aggregate_ids", ()), result=result,
        )

    def evaluate(self, components, model=None, **options):
        """Run synchronously and return the task's result, including its completion decision."""
        return self._evaluate(components, model, asynchronous=False, **options)

    def evaluate_async(self, components, model=None, *, planned=None, **options):
        """Freeze this model now, measure it concurrently, and return a Future.

        In a train/evaluate plan this implements the evaluation stage. For a
        validation used only by the trainer, pass planned=False.
        """
        if planned is None:
            planned = "evaluate" in self.record["stages"]
        # Enqueue the first request before any duplicate can queue a waiter
        # ahead of it on the same bounded executor.
        with self._lock:
            return self._evaluate(components, model, asynchronous=True, planned=planned, **options)

    def _evaluate(self, components, model, *, component="evaluator", predictions=None,
                  progress=None, checkpoint=None, selection=None, metric=None,
                  request_id=None, relation=None, source_evaluations=(), reason=None,
                  planned=False, asynchronous=False):
        if planned and "evaluate" not in self.record["stages"]:
            raise ValueError("Planned evaluation requires an evaluate stage")
        if model is not None and predictions is not None:
            raise ValueError("An evaluation accepts only one model or prediction target")
        progress = deepcopy(progress)
        target = {"component": component, "progress": progress, "selection": selection, "metric": metric}
        models = None
        if model is not None:
            models = model if isinstance(model, Mapping) else {getattr(self, "_model_name", "model"): model}
            target["models"] = list(sorted(models))
        copy_id = None
        if checkpoint:
            if checkpoint == self.record.get("initial_checkpoint") and model is None:
                copy_id = self.record.get("input_checkpoint_artifact")
            with self.store.transaction() as db:
                item = self.store.artifacts.get(db, checkpoint)
            target.update(checkpoint_id=checkpoint, source_run=item["run_id"])
            if progress is None:
                progress = target["progress"] = item["details"].get("progress")
            if metric is None:
                target["metric"] = item["details"].get("metric")
        predictions_id = self.record.get("input_predictions_artifact") if planned and not checkpoint else None
        if predictions_id:
            target["input_predictions_artifact"] = predictions_id
            original = self.record.get("prediction_artifact")
            if original:
                target["prediction_artifact"] = original
                with self.store.transaction() as db:
                    target["source_run"] = self.store.artifacts.get(db, original)["run_id"]
            else:
                target["external_input"] = True
        if predictions is not None:
            predictions = bytes(memoryview(predictions))
            target["predictions_sha256"] = hashlib.sha256(predictions).hexdigest()
        spec = components.specs[component]
        conditions = {"component": component, **spec, "split": spec["settings"].get("split"),
                      "source_id": self.record.get("source_id"), "versions": self.record.get("versions", {})}

        def reserve(db, eid):
            if checkpoint and not copy_id:
                self.store.artifacts.acquire(db, checkpoint, f"{self.run_id}:{eid}")

        eid, created = self.history.begin_evaluation(
            self.run_id, target, conditions, request_id=request_id, relation=relation,
            source_evaluations=source_evaluations, reason=reason, on_create=reserve,
        )
        self.last_evaluation = eid
        # Reserve first, even if state_dict() crashes the worker. Freeze before
        # returning to a trainer that may immediately mutate the live weights.
        snapshot = None
        preparation_error = None
        if models is not None:
            try:
                if created or self._has_model_fingerprint(eid):
                    snapshot = encode_states({k: m.state_dict() for k, m in models.items()}, {}, progress or {})
                    self.history.identify_model(self.run_id, eid, hashlib.sha256(snapshot).hexdigest())
            except Exception as exc:
                if not created:
                    raise
                preparation_error = exc

        def action():
            if not created:
                return self._existing_result(eid)
            if preparation_error is not None:
                failure = str(preparation_error) or type(preparation_error).__name__
                self.history.finish_evaluation(self.run_id, eid, "failed", reason=failure)
                if planned:
                    self._evaluation_failure("failed", failure)
                if checkpoint and not copy_id:
                    with self.store.transaction(write=True) as db:
                        self.store.artifacts.release(db, f"{self.run_id}:{eid}")
                raise preparation_error
            live_model = None if asynchronous else model
            return self._measure(components, component, eid, progress, planned, checkpoint, copy_id,
                                 predictions_id, predictions, live_model, snapshot)

        if not asynchronous:
            return action()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(thread_name_prefix="automl-evaluate")
        future = self._executor.submit(action)
        future.evaluation_id = eid
        if planned:
            self._planned_futures.append(future)
        return future

    def _has_model_fingerprint(self, evaluation_id):
        """A duplicate compares against the first caller's frozen weights, if available."""
        while True:
            record = self.history.get_evaluation(self.run_id, evaluation_id)
            if "model_sha256" in record["target"]:
                return True
            if record["state"] != "running":
                return False
            time.sleep(0.03)

    def _existing_result(self, evaluation_id):
        while True:
            record = self.history.get_evaluation(self.run_id, evaluation_id)
            if record["state"] == "interrupted":
                raise Interrupted(record["reason"])
            if record["state"] != "running":
                if "result" in record:
                    return {**record["result"], "evaluation_id": evaluation_id}
                raise ExperimentError("execution_failed", record.get("reason") or "Evaluation failed",
                                      evaluation_id=evaluation_id)
            time.sleep(0.03)

    def _save_evaluation_start(self, eid, checkpoint, lease, copy_id, named_models, progress,
                               predictions_id, predictions):
        with self.store.transaction(write=True) as db:
            artifact_id = predictions_id
            details = {"stage": "evaluate", "evaluation_id": eid, "checkpoint_id": checkpoint}
            if named_models:
                actual = encode_states({k: m.state_dict() for k, m in named_models.items()}, {}, progress or {})
                artifact_id = self.store.artifacts.save(db, self.run_id, "start_state", {"states": actual},
                                                        details=details)
            elif predictions is not None and artifact_id is None:
                artifact_id = self.store.artifacts.save(db, self.run_id, "input_predictions",
                                                        {"predictions": predictions}, details=details)
            if artifact_id is not None:
                self.store.event(db, self.run_id, "start_state_saved", {**details, "artifact_id": artifact_id})
            if self.store.run(db, self.run_id)["stages"]["evaluate"] == "pending":
                self.store.stage(db, self.run_id, "evaluate", "running")
            self.store.artifacts.release(db, lease)
            if copy_id:
                self.store.artifacts.release(db, self.run_id, checkpoint)

    def _measure(self, components, component, eid, progress, planned, checkpoint, copy_id,
                 predictions_id, predictions, live_model, snapshot):
        child = RunContext(self.store, self.run_id, evaluation_id=eid, progress=progress)
        lease = f"{self.run_id}:{eid}"
        try:
            if self.stop_requested():
                raise Interrupted("Evaluation stopped before measuring")
            if checkpoint and snapshot is None:
                with self.store.transaction() as db:
                    snapshot = self.store.artifacts.read(db, copy_id or checkpoint, "states")
            named = {}
            model = live_model
            if live_model is not None:
                named = live_model if isinstance(live_model, Mapping) else {getattr(self, "_model_name", "model"): live_model}
            elif snapshot is not None:
                states = decode_states(snapshot)
                named = {k: components.create(k, Model, mode="restore", state=s)
                         for k, s in states["models"].items()}
                model = next(iter(named.values())) if len(named) == 1 else named
            if predictions_id:
                with self.store.transaction() as db:
                    predictions = self.store.artifacts.read(db, predictions_id, "predictions")

            # Release the source as soon as all future reads use the independent copy.
            if planned:
                self._save_evaluation_start(eid, checkpoint, lease, copy_id, named, progress,
                                            predictions_id, predictions)
            elif checkpoint:
                with self.store.transaction(write=True) as db:
                    self.store.artifacts.release(db, lease)
            evaluator = components.create(component, Evaluator)
            result = evaluate(evaluator, components, child, model, predictions=predictions)
            while True:
                try:
                    report = child.finish_evaluation(result)
                    break
                except ExperimentError as exc:
                    if not planned or not exc.details.get("missing_aggregate_ids"):
                        raise
                    # A writer can still be publishing the final aggregate.
                    # Keep the planned stage running until that receipt exists.
                    if self.stop_requested():
                        raise Interrupted("Stopped while waiting for required evaluation aggregates")
                    time.sleep(0.1)
            if planned and not report["complete"]:
                self._evaluation_failure("failed", report["reason"])
            return {**result, "evaluation_id": eid}
        except Exception as exc:
            if isinstance(exc, ExperimentError) and exc.details.get("missing_aggregate_ids"):
                raise
            state = "interrupted" if isinstance(exc, Interrupted) else "failed"
            failure = str(exc) or type(exc).__name__
            self.history.finish_evaluation(self.run_id, eid, state, reason=failure)
            if planned:
                self._evaluation_failure(state, failure)
            raise
        finally:
            if checkpoint:
                with self.store.transaction(write=True) as db:
                    self.store.artifacts.release(db, lease)
                    if copy_id:
                        self.store.artifacts.release(db, self.run_id, checkpoint)

    def _evaluation_failure(self, state, reason):
        with self.store.transaction(write=True) as db:
            if state == "interrupted" and self._executor is not None:
                # Another evaluation can still fail before all threads stop.
                return
            if self.store.run(db, self.run_id)["stages"]["evaluate"] in ("pending", "running"):
                self.store.stage(db, self.run_id, "evaluate", state, reason=reason)

    def join_evaluations(self):
        """Wait for in-flight work even after training failed or safely stopped."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        interrupted = None
        for future in self._planned_futures:
            try:
                result = future.result()
                if not result["complete"]:
                    self._evaluation_failure("failed", result.get("incomplete_reason") or "Incomplete evaluation")
            except Interrupted as exc:
                interrupted = str(exc) or "Evaluator reached a safe stopping point"
            except Exception as exc:
                traceback.print_exc()
                self._evaluation_failure("failed", str(exc) or type(exc).__name__)
        if interrupted:
            self._evaluation_failure("interrupted", interrupted)
        return bool(self._planned_futures)
