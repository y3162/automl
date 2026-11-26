"""Execute the accepted plan and report each stage independently."""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping

from automl.trainer import Trainer

from .config import Components, load_implementation
from .errors import ExperimentError
from .runtime import Interrupted, RunContext


def _verify_usage(context, usage_range):
    """Late usage reports may still arrive from a distributed training process."""
    while True:
        try:
            context.verify_usage(usage_range)
            return
        except ExperimentError as exc:
            if not exc.details.get("missing"):
                raise
        if context.stop_requested():
            raise Interrupted("Stopped while waiting for training data usage")
        time.sleep(0.1)


def _train(context, components):
    try:
        if context.stop_requested():
            raise Interrupted("Training stopped before its first update")
        method = "resume" if context.record["mode"] == "resume" else "new"
        trainer = components.create("trainer", Trainer)
        with context.store.transaction(write=True) as db:
            context.store.update_run(db, context.run_id, trainer_supports_resume=bool(getattr(trainer, "supports_resume", False)))
        result = trainer.train(components, context, start_method=method)
        if not isinstance(result, Mapping):
            raise ValueError("Trainer must return its training result as a mapping")
        if not context.store.status(context.run_id)["training_started"]:
            raise ValueError("Trainer returned without saving its starting state")
        if result.get("status") == "interrupted":
            raise Interrupted(result.get("reason") or "Trainer reported a safe stop")
        if result.get("status") == "failed" or result.get("succeeded") is False:
            raise ValueError(result.get("reason") or "Trainer reported failure")
        _verify_usage(context, result.get("usage_range"))
        with context.store.transaction(write=True) as db:
            context.store.event(db, context.run_id, "training_succeeded", {"result": dict(result)})
            context.store.stage(db, context.run_id, "train", "succeeded", reason=result.get("reason"))
        return result
    except Interrupted as exc:
        context.stage("train", "interrupted", str(exc) or "Trainer reached a safe stopping point")
    except Exception as exc:
        traceback.print_exc()
        context.stage("train", "failed", str(exc) or type(exc).__name__)
    return None


def _evaluate_stage(context, components, train_result, parallel):
    with context.store.transaction() as db:
        record = context.store.run(db, context.run_id)
    stages = record["stages"]
    if stages["evaluate"] in ("failed", "interrupted"):
        return
    if parallel:
        context.stage("evaluate", "succeeded")
        return
    if stages.get("train") in ("interrupted", "failed"):
        reason = "前段の学習が中断されたため" if stages["train"] == "interrupted" else "前段の学習が失敗したため"
        context.stage("evaluate", "skipped", reason)
        return
    try:
        checkpoint = record.get("initial_checkpoint")
        reference = record.get("checkpoint_reference")
        selection = ("explicit" if reference == checkpoint else "latest") if checkpoint else None
        if train_result is not None:
            checkpoint = train_result.get("checkpoint")
            selection = train_result.get("selection", "trainer")
            if not checkpoint:
                state = train_result.get("evaluation_disposition")
                reason = train_result.get("evaluation_reason")
                if state not in ("skipped", "failed") or not reason:
                    raise ValueError("Trainer selected no checkpoint and did not report a disposition and reason")
                context.stage("evaluate", state, reason)
                return
        spec = components.specs["evaluator"]
        evaluator = load_implementation(spec["implementation"])
        lineage = evaluator.evaluation_lineage(spec["settings"])
        result = context.evaluate(components, checkpoint=checkpoint, selection=selection, planned=True, **lineage)
        if result["complete"]:
            context.stage("evaluate", "succeeded")
    except Interrupted as exc:
        context._evaluation_failure("interrupted", str(exc) or "Evaluator reached a safe stopping point")
    except Exception as exc:
        traceback.print_exc()
        context._evaluation_failure("failed", str(exc) or type(exc).__name__)


def execute(store, run_id, *, source, tree):
    """The supervisor invokes this in the isolated source tree's Python process."""
    context = RunContext(store, run_id)
    record = context.record
    versions = {**record.get("versions", {}), **record.get("dataset_versions", {})}
    components = Components(record["config"], source, versions=versions)
    result = None
    try:
        if "train" in record["stages"]:
            result = _train(context, components)
    finally:
        parallel = context.join_evaluations()
    if "evaluate" in record["stages"]:
        _evaluate_stage(context, components, result, parallel)
