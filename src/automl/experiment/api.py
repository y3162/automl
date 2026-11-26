"""Experiment admission, lineage and public project operations."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import engine, replay
from .config import load_config, validate_config
from .errors import ExperimentError
from .evaluations import EvaluationHistory
from .state import decode_states
from .store import Store


@dataclass
class _Inputs:
    checkpoint: dict | None
    parent: dict | None
    prediction: dict | None

    def identities(self):
        return tuple(record["id"] if record else None
                     for record in (self.checkpoint, self.parent, self.prediction))


@dataclass
class _RunPlan:
    config: dict
    manifest: dict
    tree: Path
    versions: dict


def _require(condition, message, **details):
    if not condition:
        raise ExperimentError("invalid_request", message, **details)


class Experiment(EvaluationHistory):
    """Run and inspect experiments in a transferable store rooted at `root`.

    The project at `source` supplies new code and resolves relative data paths.
    Workers use a frozen copy of the code that was validated at admission.
    """

    def __init__(self, root, *, source=None, read_only=False):
        self.root = Path(root).resolve()
        self.source = Path.cwd() if source is None else Path(source).resolve()
        super().__init__(Store(self.root, read_only=read_only))

    def branch(self, origin_run=None, *, checkpoint=None, request_id=None):
        """Create a branch and optionally record its originating run and checkpoint."""
        args = {"origin_run": origin_run, "checkpoint": checkpoint}
        with self.store.transaction(write=True) as db:
            previous = self.store.request(db, request_id, "branch", args)
            if previous is not None:
                return previous
            branch = self.store.create_branch(db, origin_run=origin_run, origin_checkpoint=checkpoint)
            self.store.receipt(db, request_id, "branch", args, branch)
            return branch

    def start(self, branch=None, config=None, *, initialize=None, evaluate=None,
              predictions=None, prediction_artifact=None, retain=None, fork=None,
              current_source=False, request_id=None):
        """Accept a run and launch it in the background.

        An existing branch returns a run ID. A fork returns both IDs.
        Supply a request ID to retrieve the same result on a retry.
        """
        if retain is not None:
            _require(isinstance(retain, (list, tuple)) and all(isinstance(n, str) and n for n in retain),
                     "retain requires a sequence of model parameter names")
        args = dict(branch=branch, config=str(config) if config is not None else None,
                    initialize=initialize, evaluate=evaluate,
                    predictions=str(predictions) if predictions is not None else None,
                    prediction_artifact=prediction_artifact, retain=None if retain is None else list(retain),
                    fork=fork, current_source=current_source)
        return self._submit("start", args, request_id)

    def resume(self, branch, *, request_id=None):
        """Continue from the latest usable checkpoint of the last run that trained."""
        return self._submit("resume", {"branch": branch}, request_id)

    def _select(self, db, args, resume):
        branch = args["branch"]
        ref = args.get("fork") or args.get("initialize") or args.get("evaluate")
        parent = prediction = origin = None
        if branch is not None:
            origin = self.store.branch(db, branch)["origin_run"]
            active = self.store.active_run(db, branch)
            if active:
                raise ExperimentError("branch_busy", "Branch already has an active run", active_run_id=active)
        if resume:
            parent = self.store.last_training(db, branch)
            _require(parent, "This branch has no run that started training", branch=branch)
            ref = parent["id"]
        checkpoint = self.store.artifacts.resolve(db, ref) if ref is not None else None
        if checkpoint:
            parent = self.store.run(db, checkpoint["run_id"])
            if args.get("evaluate") and origin:
                prior = self.store.run(db, origin)
                if prior.get("mode") == "evaluate" and prior.get("initial_checkpoint") == checkpoint["id"]:
                    parent = prior
        if args.get("prediction_artifact") is not None:
            prediction = self.store.artifacts.get(db, args["prediction_artifact"])
            _require(prediction["kind"] in {"predictions", "prediction", "input_predictions", "artifact", "output"}
                     and len(prediction["files"]) == 1, "Expected a prediction file",
                     artifact_id=prediction["id"])
        return _Inputs(checkpoint, parent, prediction)

    def _submit(self, op, args, request_id):
        resume = op == "resume"
        _require(not self.store.read_only, "Experiment was opened read-only")
        with self.store.transaction() as db:
            previous = self.store.request(db, request_id, op, args)
            if previous is not None:
                return previous
            if not resume:
                self._check_arguments(args)
            active = self.store.active_run(db, args["branch"])
        if active:
            self.status(active, reconcile=True)

        # Plugins can be slow. Validate outside the write lock and resolve
        # mutable references again in the transaction reserving the run.
        while True:
            with self.store.transaction() as db:
                previous = self.store.request(db, request_id, op, args)
                if previous is not None:
                    return previous
                inputs = self._select(db, args, resume)
            plan = self._prepare(args, inputs, resume)
            predictions = self._prediction_bytes(args, inputs.prediction)
            with self.store.transaction(write=True) as db:
                previous = self.store.request(db, request_id, op, args)
                if previous is not None:
                    return previous
                selection = self._select(db, args, resume)
                if selection.identities() != inputs.identities():
                    continue
                run, branch = self._reserve(db, args, inputs, resume, plan, predictions)
                result = {"run_id": run, "branch_id": branch} if args.get("fork") else run
                self.store.receipt(db, request_id, op, args, result)
                break
        self._launch(run, inputs.checkpoint, plan)
        return result

    @staticmethod
    def _check_arguments(args):
        _require((args["branch"] is None) != (args["fork"] is None), "Specify exactly one of branch and fork")
        targets = ("evaluate", "prediction_artifact", "predictions", "initialize", "fork")
        _require(sum(args[key] is not None for key in targets) <= 1,
                 "Specify a single initialization or evaluation input")
        for key in targets:
            if args[key] is not None:
                _require(isinstance(args[key], str) and args[key], f"Expected a nonempty {key} reference")
        _require(not args["retain"] or args["initialize"] or args["fork"],
                 "retain requires a checkpoint for initialization")
        _require(not args["current_source"] or args["evaluate"], "current_source requires checkpoint evaluation")
        _require(args["config"] is not None or args["evaluate"], "A configuration file is required")

    def _prepare(self, args, inputs, resume):
        parent = inputs.parent
        evaluate = args.get("evaluate") is not None
        inherit = resume or evaluate
        parent_manifest = replay.read_manifest(self.root, parent["id"]) if inherit else None
        if inherit and parent.get("source_id"):
            _require(parent_manifest is not None, "The recorded replay manifest is missing", run_id=parent["id"])
            expected = parent.get("manifest_sha256")
            _require(parent_manifest.get("source_id") == parent["source_id"]
                     and (expected is None or replay.manifest_digest(parent_manifest) == expected),
                     "The recorded replay manifest has changed", run_id=parent["id"])
        base = parent["config"] if inherit else None
        if evaluate:
            base = {**base, "plan": ["evaluate"]}
        raw = load_config(args.get("config"), self.source, base=base)
        if not resume and raw.get("plan") in (["train"], ["evaluate"], ["train", "evaluate"]):
            training = "train" in raw["plan"]
            has_target = any(args.get(key) is not None for key in ("evaluate", "predictions", "prediction_artifact"))
            _require(not training or not has_target, "An evaluation input requires an evaluation-only plan")
            _require(training or not (args.get("initialize") or args.get("fork")),
                     "Model initialization requires a training plan")
        if parent_manifest and not args.get("current_source"):
            source_id = parent_manifest["source_id"]
        else:
            source_id = replay.capture_source(self.root, self.source)
        tree = replay.restore_source(self.root, source_id)
        if resume and parent_manifest:
            issues = replay.verify_replay(self.root, parent["id"], self.source,
                                          versions=parent.get("dataset_versions", {}),
                                          expected_manifest=parent.get("manifest_sha256"))
            _require(not issues, "Cannot reproduce the training conditions", issues=issues)

        options = {}
        if inherit:
            with self.store.transaction() as db:
                data = self.store.artifacts.read(db, inputs.checkpoint["id"], "states")
            states = decode_states(data)
            if resume:
                options["resume_state"] = states
            elif isinstance(raw.get("components"), dict):
                options["models"] = {
                    name: state for name, state in states["models"].items()
                    if args["current_source"] or raw["components"].get(name) != base["components"].get(name)
                }
        if (raw.get("plan") == ["evaluate"] and not evaluate and
                args.get("prediction_artifact") is None and args.get("predictions") is None):
            options["evaluation_input"] = "measurements"
        config, files = validate_config(raw, self.source, code_source=tree, store_root=self.root, **options)
        manifest = replay.build_manifest(source_id, config, files, root=self.root, source=self.source)
        if inherit and not parent_manifest:
            manifest["unrecorded_parent_source"] = parent["id"]
        versions = {}
        if parent:
            versions = {name: version for name, version in parent.get("dataset_versions", {}).items()
                        if config["components"].get(name) == parent["config"]["components"].get(name)}
        return _RunPlan(config, manifest, tree, versions)

    def _prediction_bytes(self, args, prediction):
        if prediction:
            with self.store.transaction() as db:
                return self.store.artifacts.read(db, prediction["id"], next(iter(prediction["files"])))
        if args.get("predictions") is not None:
            try:
                return (self.source / args["predictions"]).read_bytes()
            except OSError as exc:
                raise ExperimentError("invalid_request", f"Cannot read predictions: {exc}",
                                      path=args["predictions"]) from exc
        return None

    def _reserve(self, db, args, inputs, resume, plan, predictions):
        checkpoint, parent, prediction = inputs.checkpoint, inputs.parent, inputs.prediction
        config = plan.config
        branch = args["branch"]
        checkpoint_id = checkpoint["id"] if checkpoint else None
        if args.get("fork"):
            branch = self.store.create_branch(db, origin_run=parent["id"], origin_checkpoint=checkpoint_id)
        training = "train" in config["plan"]
        if training and not resume:
            _require(self.store.last_training(db, branch) is None,
                     "Training has already started on this branch. Resume it or create a new branch.", branch=branch)
        reference = parent["id"] if resume else args.get("initialize") or args.get("fork") or args.get("evaluate")
        fields = {
            "initial_checkpoint": checkpoint_id, "checkpoint_reference": reference,
            "resume_from": parent["id"] if resume else None,
            "resume_checkpoint": checkpoint_id if resume else None,
            "mode": "resume" if resume else "train" if training else "evaluate",
            "retain": args.get("retain"), "versions": plan.versions,
            "manifest_sha256": replay.manifest_digest(plan.manifest),
            "source_id": plan.manifest["source_id"], "prediction_artifact": args.get("prediction_artifact"),
            "prediction_run": prediction["run_id"] if prediction else None,
            "predictions": args.get("predictions"), **engine.launch_owner(),
        }
        run = self.store.create_run(db, branch, config, **fields)
        if checkpoint:
            self.store.artifacts.acquire(db, checkpoint_id, run)
        if predictions is not None:
            artifact = self.store.artifacts.save(
                db, run, "input_predictions", {"predictions": predictions},
                details={"external_path": args.get("predictions"), "artifact_id": args.get("prediction_artifact"),
                         "run_id": fields["prediction_run"]})
            self.store.update_run(db, run, input_predictions_artifact=artifact)
        if args.get("evaluate"):
            previous = parent["config"]["components"]
            changed = {key: spec for key, spec in config["components"].items() if previous.get(key) != spec}
            self.store.event(db, run, "evaluation_conditions", {
                "origin_run": parent["id"], "checkpoint": checkpoint_id, "changes": changed,
                "previous_source": parent.get("source_id"), "source": plan.manifest["source_id"]})
        return run, branch

    def _launch(self, run, checkpoint, plan):
        try:
            replay.write_manifest(self.root, run, plan.manifest)
            directory = self.root / "runs" / run
            (directory / "config.yaml").write_text(yaml.safe_dump(plan.config, sort_keys=False), encoding="utf-8")
            if checkpoint:
                checkpoint_id = checkpoint["id"]
                with self.store.transaction() as db:
                    states = self.store.artifacts.read(db, checkpoint_id, "states")
                with self.store.artifacts.stage(run, "input_checkpoint", {"states": states},
                                                details={"checkpoint_id": checkpoint_id}) as publish:
                    with self.store.transaction(write=True) as db:
                        if self.store.run(db, run)["status"] != "accepted":
                            return
                        self.store.update_run(db, run, input_checkpoint_artifact=publish(db))
                        self.store.artifacts.release(db, run, checkpoint_id)
            engine.launch(self.store, run, tree=plan.tree, source=self.source)
        except Exception as exc:
            with self.store.transaction(write=True) as db:
                engine.fail_launch(self.store, db, run, str(exc))

    def stop(self, run):
        engine.stop(self.store, run)

    def status(self, run, *, reconcile=False):
        return engine.status(self.store, run, reconcile=reconcile and not self.store.read_only)

    def wait(self, run, *, timeout=None):
        if timeout is not None:
            _require(isinstance(timeout, (int, float)) and math.isfinite(timeout) and timeout >= 0,
                     "timeout must be a finite, nonnegative number of seconds")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            record = self.status(run, reconcile=True)
            if record["status"] not in ("accepted", "running"):
                return record
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return record
            time.sleep(0.1 if remaining is None else min(remaining, 0.1))

    def delete_checkpoint(self, checkpoint_id):
        return self.store.artifacts.delete(checkpoint_id)

    def history(self, run=None, *, summary=False):
        return self.store.history(run, summary=summary)

    def tracking_revisions(self):
        return self.store.tracking_revisions()

    def tracking_snapshot(self, run):
        return self.store.tracking_snapshot(run)

    def verify(self, run=None):
        report = self.store.verify(run)
        ids = [run] if run is not None else self.tracking_revisions()
        unreadable = {issue["run_id"] for issue in report["issues"] if issue.get("item") == "run_record"}
        for run_id in ids:
            if run_id in unreadable:
                continue
            record = self.store.status(run_id)
            if run is not None or record.get("source_id"):
                issues = replay.verify_replay(self.root, run_id, self.source, check_environment=run is not None,
                                              versions=record.get("dataset_versions", {}),
                                              expected_manifest=record.get("manifest_sha256"))
                report["issues"].extend(issues)
        if run is not None:
            report["ready"] = not report["issues"]
        return report
