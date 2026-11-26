"""Exercise the public API and detached CLI workers against a transferable store."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import shutil
import subprocess
import sys
import time

import pytest
import yaml

from automl.experiment import Experiment, ExperimentError, engine


PROJECT = Path(__file__).resolve().parents[3]
MOCK = PROJECT / "data" / "mock"
INLINE = "configs/examples/xor/inline.yaml"
TERMINAL = {"succeeded", "failed", "interrupted"}


@pytest.fixture
def workspace():
    store_parent = MOCK / "store"
    store_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="xor-api-", dir=store_parent) as scratch:
        work = MOCK / "work" / Path(scratch).name
        source = work / "project"
        shutil.copytree(PROJECT / "configs" / "examples" / "xor", source / "configs" / "examples" / "xor")
        fast = {"extends": INLINE, "components": {"trainer": {"settings": {"rounds": 2, "save_every": 1}}}}
        (source / "fast.yaml").write_text(yaml.safe_dump(fast))
        try:
            yield Path(scratch) / "original", source
        finally:
            shutil.rmtree(work, ignore_errors=True)


def wait(experiment, run, *, until=None, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        state = experiment.status(run, reconcile=True)
        if state["status"] in TERMINAL or (until and until()):
            return state
        assert time.monotonic() < deadline, f"Run {run} did not reach the expected state: {state}"
        time.sleep(0.05)


def cli(root, source, *args, exit_code=0):
    proc = subprocess.run(
        [sys.executable, "-P", "-m", "automl", "--root", str(root), *args],
        env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        cwd=source, capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == exit_code, (proc.stdout, proc.stderr)
    return json.loads(proc.stdout or proc.stderr)


def test_cancel_before_launch_then_retry_training(workspace):
    root, source = workspace
    exp = Experiment(root, source=source)
    branch = exp.branch(request_id="branch-request")
    assert exp.branch(request_id="branch-request") == branch
    original_launch = engine.launch

    def cancel_at_handoff(store, run, **kwargs):
        exp.stop(run)
        original_launch(store, run, **kwargs)

    with patch.object(engine, "launch", cancel_at_handoff):
        cancelled = exp.start(branch, "fast.yaml", request_id="cancel")
    assert exp.status(cancelled)["status"] == "interrupted"
    assert exp.start(branch, "fast.yaml", request_id="cancel") == cancelled
    retry = exp.start(branch, "fast.yaml", request_id="retry")
    state = wait(exp, retry)
    assert state["status"] == "succeeded", state.get("worker_error") or state.get("error")
    assert [r["id"] for r in exp.history(summary=True)["runs"]] == [cancelled, retry]
    with pytest.raises(ExperimentError) as conflict:
        exp.start(branch, INLINE, request_id="retry")
    assert conflict.value.code == "invalid_request"
    assert exp.verify()["issues"] == []


def test_fork_with_empty_retain_trains_the_csv_product_model_from_scratch(workspace):
    root, source = workspace
    (source / "one-round.yaml").write_text(yaml.safe_dump({
        "extends": "configs/examples/xor/train.yaml",
        "components": {"trainer": {"settings": {"rounds": 1}}},
    }))
    exp = Experiment(root, source=source)
    parent = exp.start(exp.branch(), "one-round.yaml")
    assert wait(exp, parent)["status"] == "succeeded"

    child = exp.start(config="configs/examples/xor/product_target.yaml", fork=parent, retain=[])["run_id"]
    status = wait(exp, child)
    assert status["status"] == "succeeded", status["stage_reasons"]
    assert status["origin_run"] == parent
    assert status["stage_reasons"]["train"] == "target_score_reached"
    history = exp.history(child)
    initialized, = (e["payload"] for e in history["events"] if e["kind"] == "model_initialized")
    assert initialized["retained"] == []
    assert initialized["initialized"] == ["bias", "weight"]
    checkpoint, = (a for a in history["artifacts"] if a["kind"] == "checkpoint")
    assert checkpoint["order"] == 1 and checkpoint["details"]["score"] == 1.0
    assert exp.verify(child)["ready"]


def test_stop_then_resume_from_the_saved_training_state(workspace):
    root, source = workspace
    # Gate only the first training process after a durable checkpoint. This
    # exercises the real stop request without racing a short XOR experiment.
    (source / "gated.py").write_text('''
import time
from examples.xor.components.learning import MiniBatch

class PauseAfterSave:
    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        return getattr(self.context, name)

    def save_checkpoint(self, **kwargs):
        saved = self.context.save_checkpoint(**kwargs)
        deadline = time.monotonic() + 60
        while not self.context.stop_requested():
            if time.monotonic() > deadline:
                raise TimeoutError("test did not request stop")
            time.sleep(0.03)
        return saved

class GatedTrainer(MiniBatch):
    def train(self, components, context, *, start_method):
        if start_method == "new":
            context = PauseAfterSave(context)
        return super().train(components, context, start_method=start_method)
''')
    (source / "gated.yaml").write_text(yaml.safe_dump({
        "extends": INLINE,
        "components": {"trainer": {"implementation": "gated.GatedTrainer", "settings": {
            "rounds": 3, "save_every": 1, "selection_metric": "error_rate", "minimize": True,
        }}},
    }))
    exp = Experiment(root, source=source)
    branch = exp.branch()
    run = exp.start(branch, "gated.yaml")
    def checkpoints():
        return [a for a in exp.history(run)["artifacts"] if a["kind"] == "checkpoint"]

    state = wait(exp, run, until=checkpoints)
    assert state["status"] == "running", state
    try:
        with pytest.raises(ExperimentError) as busy:
            exp.resume(branch)
        assert busy.value.code == "branch_busy"
        assert busy.value.details["active_run_id"] == run
    finally:
        exp.stop(run)
    stopped = wait(exp, run)
    assert stopped["status"] == "interrupted", stopped
    assert stopped["stages"] == {"train": "interrupted", "evaluate": "skipped"}
    checkpoint = max(checkpoints(), key=lambda a: a["order"])["id"]

    # The worker and preflight for resume must use the saved project code.
    (source / "gated.py").write_text("raise RuntimeError('live file must not be imported on resume')")
    resumed = exp.resume(branch, request_id="resume-request")
    final = wait(exp, resumed)
    assert final["status"] == "succeeded", final.get("worker_error") or final.get("error")
    assert final["resume_from"] == run
    assert final["resume_checkpoint"] == checkpoint
    assert exp.resume(branch, request_id="resume-request") == resumed
    assert exp.verify(resumed)["ready"]


def test_cli_fork_backup_cursor_and_checkpoint_deletion(workspace):
    root, source = workspace
    branch = cli(root, source, "branch")["branch_id"]
    started = cli(root, source, "start", "fast.yaml", "--branch", branch, "--request-id", "start")
    run = started["run_id"]
    assert started == {"branch_id": branch, "run_id": run, "origin_run": None,
                       "origin_checkpoint": None, "initial_checkpoint": None}
    assert cli(root, source, "wait", run, "--timeout", "60")["status"] == "succeeded"
    exp = Experiment(root, source=source)
    artifacts = exp.history(run)["artifacts"]
    cp = max((a for a in artifacts if a["kind"] == "checkpoint"), key=lambda a: a["order"])["id"]

    def fork(_):
        return Experiment(root, source=source).start(config="fast.yaml", fork=cp, request_id="fork")

    with ThreadPoolExecutor(max_workers=2) as pool:
        child, duplicate = pool.map(fork, range(2))
    assert child == duplicate
    assert child["branch_id"] != branch
    state = wait(exp, child["run_id"])
    assert state["status"] == "succeeded", state.get("worker_error") or state.get("error")
    assert (state["origin_run"], state["initial_checkpoint"]) == (run, cp)
    assert len(exp.history(summary=True)["runs"]) == 2

    e = cli(root, source, "list-evaluations", run, "--limit", "20")["evaluations"][-1]["evaluation_id"]
    decision = cli(root, source, "record-decision", run, e, "compare on another branch",
                   "--aggregate-id", "error_rate", "--followup-run", child["run_id"])
    uses = cli(root, source, "list-uses", run, e, "--aggregate-id", "error_rate", "--limit", "5")
    assert decision["decision_id"] in {use.get("decision_id") for use in uses["uses"]}
    first_page = cli(root, source, "read-results", run, e, "--limit", "1")
    cursor = first_page["next_cursor"]
    assert cursor
    expected = exp.read_results(run, e, limit=3, cursor=cursor)

    backup = root.with_name("restored")
    clone = source.with_name("clone")
    shutil.copytree(root, backup)
    shutil.copytree(source, clone)
    root.rename(root.with_name("offline-store"))
    source.rename(source.with_name("offline-source"))
    moved = Experiment(backup, source=clone)
    assert cli(backup, clone, "read-results", run, e, "--limit", "3", "--cursor", cursor) == expected
    assert moved.delete_checkpoint(cp) == {"checkpoint_id": cp, "result": "deleted"}
    assert cli(backup, clone, "delete-checkpoint", cp)["result"] == "deleted"
    assert moved.verify()["issues"] == []
    assert moved.verify(child["run_id"])["ready"]
    closed = [a for a in moved.history(run)["artifacts"] if a["id"] == cp]
    assert closed[0]["checkpoint_state"] == "deleted"

    # Retrying the successful fork returns its original IDs even after deletion.
    assert moved.start(config="fast.yaml", fork=cp, request_id="fork") == child
    with pytest.raises(ExperimentError) as deleted:
        moved.start(branch, evaluate=cp)
    assert deleted.value.code == "invalid_request"
    assert deleted.value.details["checkpoint_id"] == cp
    assert deleted.value.details["checkpoint_state"] == "deleted"
    assert len(moved.history(summary=True)["runs"]) == 2

    # A retained consumer copy must still pass its own checksum after the
    # producer checkpoint is deleted. Damage the backup's copy to test this.
    copied = next(a for a in moved.history(child["run_id"])["artifacts"] if a["kind"] == "input_checkpoint")
    file = next(iter(copied["files"].values()))
    with (backup / file["path"]).open("ab") as stream:
        stream.write(b"corrupt")
    verification = cli(backup, clone, "verify", exit_code=1)
    assert any(issue.get("artifact_id") == copied["id"] for issue in verification["issues"])
    assert not cli(backup, clone, "verify", "--run", child["run_id"], exit_code=1)["ready"]


def test_replayed_prediction_request_uses_accepted_file_contents(workspace):
    root, source = workspace
    exp = Experiment(root, source=source)
    branch = exp.branch()
    (source / "score.yaml").write_text(yaml.safe_dump({"extends": INLINE, "plan": ["evaluate"]}))
    predictions = source / "predictions.json"
    predictions.write_text(json.dumps({"test/0": 0, "test/1": 1, "test/2": 1, "test/3": 0}))
    args = {"branch": branch, "config": "score.yaml", "predictions": predictions, "request_id": "external"}
    run = exp.start(**args)
    predictions.write_text("{}")
    assert exp.start(**args) == run
    state = wait(exp, run)
    assert state["status"] == "succeeded", state.get("worker_error") or state.get("error")
    evaluation = exp.list_evaluations(run, limit=1)["evaluations"][0]["evaluation_id"]
    aggregate = exp.get_aggregate(run, evaluation, "error_rate")
    assert aggregate["value"] == 0
    assert exp.start(**args) == run
    assert len(exp.history(summary=True)["runs"]) == 1


def test_stop_during_final_evaluation_keeps_successful_training(workspace):
    root, source = workspace
    (source / "slow_eval.py").write_text("""
import time
from examples.xor.components.scoring import ErrorRate

class WaitForStop(ErrorRate):
    def evaluate(self, components, context, model, *, predictions=None):
        deadline = time.monotonic() + 60
        while not context.stop_requested():
            if time.monotonic() > deadline:
                raise TimeoutError("test did not request stop")
            time.sleep(0.03)
        return super().evaluate(components, context, model, predictions=predictions)
""")
    (source / "training.yaml").write_text(yaml.safe_dump({
        "extends": "fast.yaml",
        "components": {"evaluator": {"implementation": "slow_eval.WaitForStop"}},
    }))
    exp = Experiment(root, source=source)
    run = exp.start(exp.branch(), "training.yaml")

    def evaluating():
        return exp.status(run)["stages"] == {"train": "succeeded", "evaluate": "running"}

    state = wait(exp, run, until=evaluating)
    assert state["status"] == "running", state.get("error")
    assert cli(root, source, "stop", run) == {"run_id": run, "stop_requested": True}
    outcome = cli(root, source, "wait", run, "--timeout", "60")
    assert outcome["status"] == "interrupted"
    assert outcome["stages"] == {"train": "succeeded", "evaluate": "interrupted"}
    assert any(a["kind"] == "checkpoint" and a["checkpoint_state"] == "retained"
               for a in exp.history(run)["artifacts"])


def test_sequential_updates_restore_both_named_xor_models(workspace):
    root, source = workspace
    # Two learning rates produce distinct, exactly known states after one pass.
    # Post-training and later independent evaluation must restore both by name.
    (source / "paired.py").write_text("""
import numpy as np
from automl.trainer import Trainer
from examples.xor.components.scoring import Accuracy

class Sequential(Trainer):
    def train(self, components, context, *, start_method):
        version, data = context.open_dataset(components)
        samples = list(data.iter_samples(split="train"))
        x = np.array([sample.data for sample in samples])
        y = np.array([sample.metadata["label"] for sample in samples])
        ids = [sample.sample_id for sample in samples]
        models = {name: context.prepare_model(components, name) for name in ("model", "second")}
        def states():
            return {name: model.state_dict() for name, model in models.items()}
        context.start_training(model_states=states(), training_state={}, progress={"step": 0})
        for step, (name, model) in enumerate(models.items(), start=1):
            model.update(x, y, rate=step)
            context.add(usage_id=name, position=step, dataset_version=version,
                        split="train", sample_ids=ids, details={"model": name})
        checkpoint = context.save_checkpoint(model_states=states(), training_state={},
                                             progress={"step": 2}, usage_range=2)
        return {"checkpoint": checkpoint, "usage_range": 2, "reason": "both_models_updated"}

class ScorePair(Accuracy):
    def evaluate(self, components, context, model, *, predictions=None):
        result = super().evaluate(components, context, model["second"])
        result["weights"] = {name: part.state_dict()["weight"] for name, part in model.items()}
        return result
""")
    (source / "paired.yaml").write_text(yaml.safe_dump({
        "extends": INLINE,
        "components": {
            "trainer": {"implementation": "paired.Sequential"},
            "second": {"implementation": "examples.xor.components.model.ProductLogistic"},
            "evaluator": {"implementation": "paired.ScorePair"},
        },
    }))
    exp = Experiment(root, source=source)
    run = exp.start(exp.branch(), "paired.yaml")
    outcome = wait(exp, run)
    assert outcome["status"] == "succeeded", outcome.get("error")
    history = exp.history(run)
    checkpoint = next(item for item in history["artifacts"] if item["kind"] == "checkpoint")
    assert checkpoint["details"]["model_names"] == ["model", "second"]
    replay = exp.start(exp.branch(), evaluate=checkpoint["id"])
    assert wait(exp, replay)["status"] == "succeeded"
    for measured_run in (run, replay):
        evaluation, = exp.list_evaluations(measured_run, limit=5)["evaluations"]
        result = evaluation["result"]
        assert result["weights"] == {"model": -0.5, "second": -1.0}
        assert result["accuracy"] == 1.0
    assert exp.verify(replay)["ready"]


def test_repeated_evaluation_uses_its_origin_and_checks_current_source(workspace):
    root, source = workspace
    exp = Experiment(root, source=source)
    model_module = source / "versioned.py"
    model = "from examples.xor.components.model import ProductLogistic\nclass Versioned(ProductLogistic):\n"
    model_module.write_text(model + "    pass\n")
    (source / "train.yaml").write_text(yaml.safe_dump({
        "extends": "fast.yaml", "plan": ["train"],
        "components": {"model": {"implementation": "versioned.Versioned"}},
    }))
    run = exp.start(exp.branch(), "train.yaml")
    trained = wait(exp, run)
    assert trained["status"] == "succeeded", trained["error"]
    cp = max((a for a in exp.history(run)["artifacts"] if a["kind"] == "checkpoint"),
             key=lambda a: a["order"])["id"]

    model_module.write_text(model + """
    def load_state_dict(self, state):
        raise ValueError("incompatible comparison model")
""")
    (source / "base-evaluation.yaml").write_text(yaml.safe_dump({"components": {"evaluator": {
        "implementation": "examples.xor.components.scoring.Accuracy",
        "settings": {"split": "validation", "min_count": 1},
    }}}))
    (source / "compare.yaml").write_text(yaml.safe_dump({
        "extends": "base-evaluation.yaml", "components": {"evaluator": {"settings": {"min_count": 2}}},
    }))
    comparison = exp.start(exp.branch(), "compare.yaml", evaluate=cp)
    state = wait(exp, comparison)
    assert state["status"] == "succeeded", state["error"]

    def conditions(run_id):
        return next(event["payload"] for event in exp.history(run_id)["events"]
                    if event["kind"] == "evaluation_conditions")

    first = conditions(comparison)
    assert first["origin_run"] == run
    assert first["source"] == first["previous_source"] == trained["source_id"]
    settings = first["changes"]["evaluator"]["settings"]
    assert (settings["split"], settings["min_count"]) == ("validation", 2)
    branch = exp.branch(origin_run=comparison)
    derived = cli(root, source, "start", "--branch", branch, "--evaluate", cp)["run_id"]
    result = wait(exp, derived)
    assert result["status"] == "succeeded", result["error"]
    assert result["config"]["components"]["evaluator"] == first["changes"]["evaluator"]
    inherited = conditions(derived)
    assert (inherited["origin_run"], inherited["changes"], inherited["source"]) == (comparison, {}, first["source"])

    args = ["start", "--branch", branch, "--evaluate", cp, "--current-source"]
    before = exp.history(summary=True)["runs"]
    rejected = cli(root, source, *args, exit_code=1)
    assert rejected["code"] == "invalid_config"
    assert "incompatible comparison model" in rejected["message"]
    assert exp.history(summary=True)["runs"] == before

    model_module.write_text(model + """
    def predict(self, features):
        return 1 - super().predict(features)
""")
    current = cli(root, source, *args)["run_id"]
    result = wait(exp, current)
    assert result["status"] == "succeeded", result["error"]
    live = conditions(current)
    assert live["previous_source"] == first["source"]
    assert live["source"] != first["source"]

    def accuracy(run_id):
        ev = exp.list_evaluations(run_id, limit=1)["evaluations"][0]
        assert ev["conditions"]["settings"] == settings
        return exp.get_aggregate(run_id, ev["evaluation_id"], "accuracy")["value"]

    assert accuracy(comparison) == accuracy(derived)
    assert accuracy(current) == 1 - accuracy(derived)


def test_project_prediction_output_can_be_scored_in_another_run(workspace):
    root, source = workspace
    exp = Experiment(root, source=source)
    (source / "exporter.py").write_text("""
import json
from examples.xor.components.scoring import Accuracy

class ExportPredictions(Accuracy):
    def evaluate(self, components, context, model, *, predictions=None):
        result = super().evaluate(components, context, model, predictions=predictions)
        _, dataset = context.open_dataset(components)
        values = {sample.sample_id: int(model.predict([sample.data])[0])
                  for sample in dataset.iter_samples(split=self.split)}
        context.save_artifact("predictions.json", json.dumps(values).encode())
        return result
""")
    (source / "produce.yaml").write_text(yaml.safe_dump({
        "extends": "fast.yaml", "components": {"evaluator": {"implementation": "exporter.ExportPredictions"}},
    }))
    producer = exp.start(exp.branch(), "produce.yaml")
    produced = wait(exp, producer)
    assert produced["status"] == "succeeded", produced["error"]
    output = next(a for a in exp.history(producer)["artifacts"]
                  if a["kind"] == "output" and "predictions.json" in a["files"])
    original = exp.get_aggregate(producer, output["details"]["evaluation_id"], "accuracy")["value"]
    (source / "score.yaml").write_text(yaml.safe_dump({
        "extends": INLINE, "plan": ["evaluate"],
        "components": {"evaluator": {"implementation": "examples.xor.components.scoring.Accuracy"}},
    }))
    scored = cli(root, source, "start", "score.yaml", "--branch", exp.branch(),
                 "--prediction-artifact", output["id"])["run_id"]
    final = wait(exp, scored)
    assert final["status"] == "succeeded", final["error"]
    ev = exp.list_evaluations(scored, limit=1)["evaluations"][0]
    assert ev["target"]["source_run"] == producer
    assert ev["target"]["prediction_artifact"] == output["id"]
    assert exp.get_aggregate(scored, ev["evaluation_id"], "accuracy")["value"] == original
    assert exp.verify(scored)["ready"]


@pytest.mark.parametrize("minimum", [4, 5])
def test_best_checkpoint_only_uses_complete_validation(workspace, minimum):
    root, source = workspace
    (source / "best.yaml").write_text(yaml.safe_dump({
        "extends": "fast.yaml",
        "components": {
            "trainer": {"settings": {"final_checkpoint": "best"}},
            "validation": {"settings": {"min_count": minimum}},
        },
    }))
    exp = Experiment(root, source=source)
    run = exp.start(exp.branch(), "best.yaml")
    state = wait(exp, run)
    assert state["status"] == "succeeded" and state["stages"]["train"] == "succeeded"
    evaluations = exp.list_evaluations(run, limit=20)["evaluations"]
    if minimum == 4:
        checkpoints = [a for a in exp.history(run)["artifacts"] if a["kind"] == "checkpoint"]
        # Validation error is tied at zero: pick the first trained checkpoint.
        assert [e["result"]["error_rate"] for e in evaluations] == [0, 0, 0]
        assert evaluations[-1]["target"]["checkpoint_id"] == checkpoints[0]["id"]
        assert state["stages"]["evaluate"] == "succeeded"
    else:
        assert len(evaluations) == 2 and all(e["result"]["provisional"] for e in evaluations)
        assert state["stages"]["evaluate"] == "skipped"
        assert "no complete validation" in state["stage_reasons"]["evaluate"]


def test_delete_during_evaluation_input_copy(workspace, monkeypatch):
    from threading import Event
    from automl.experiment import artifacts

    root, source = workspace
    experiment = Experiment(root, source=source)
    branch = experiment.branch()
    trained = experiment.start(branch, "fast.yaml")
    assert experiment.wait(trained, timeout=60)["status"] == "succeeded"
    checkpoints = [a for a in experiment.history(trained)["artifacts"] if a["kind"] == "checkpoint"]
    checkpoint = checkpoints[-1]["id"]
    copying, resume_copy = Event(), Event()
    original = artifacts._copy_content

    def pause_copy(content, destination):
        if destination.name == "states":
            copying.set()
            assert resume_copy.wait(30)
        return original(content, destination)

    monkeypatch.setattr(artifacts, "_copy_content", pause_copy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        evaluation = pool.submit(experiment.start, branch, evaluate=checkpoint)
        try:
            assert copying.wait(30)
            assert experiment.delete_checkpoint(checkpoint)["result"] == "in_use"
        finally:
            resume_copy.set()
        run = evaluation.result(timeout=60)
    assert experiment.delete_checkpoint(checkpoint)["result"] == "deleted"
    assert experiment.wait(run, timeout=60)["status"] == "succeeded"
    assert experiment.verify(run)["ready"]


def test_launch_failure_releases_branch_before_training(workspace, monkeypatch):
    root, source = workspace
    experiment = Experiment(root, source=source)
    branch = experiment.branch()

    def unavailable(*args, **kwargs):
        raise OSError("launch unavailable")

    with monkeypatch.context() as ctx:
        ctx.setattr(engine, "launch", unavailable)
        run = experiment.start(branch, "fast.yaml", request_id="first")
    failed = experiment.status(run)
    assert failed["error"]["code"] == "launch_failed"
    assert failed["stages"] == {"train": "failed", "evaluate": "skipped"}
    assert experiment.start(branch, "fast.yaml", request_id="first") == run
    retried = experiment.start(branch, "fast.yaml", request_id="next")
    assert experiment.wait(retried, timeout=60)["status"] == "succeeded"
