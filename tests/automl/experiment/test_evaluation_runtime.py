import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from automl.evaluator import Evaluator
from automl.experiment import Components, Experiment, ExperimentError, RunContext
from automl.experiment.store import Store


MOCK = Path(__file__).resolve().parents[3] / "data/mock"


class EarlyReport(Evaluator):
    def evaluate(self, components, context, model, *, predictions=None):
        context.record_result("sample", "sample", 1)
        return {"complete": True, "aggregate_ids": ["mean"]}


def config(implementation, **settings):
    return {"plan": ["evaluate"], "components": {"evaluator": {
        "implementation": implementation, "settings": settings,
    }}}


@pytest.fixture
def store():
    directory = MOCK / "store"
    directory.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=directory, prefix="eval-runtime-") as root:
        try:
            yield Store(root)
        finally:
            # A detached worker materializes its source cache beside the DB.
            shutil.rmtree(MOCK / "work" / Path(root).name, ignore_errors=True)


@pytest.fixture
def task(store):
    resolved = config(f"{__name__}.EarlyReport")
    with store.transaction(write=True) as db:
        run = store.create_run(db, store.create_branch(db), resolved)
    return RunContext(store, run), Components(resolved, root=Path.cwd())


def test_unpublished_aggregate_can_be_published_and_success_retried(task):
    context, components = task
    request = {"request_id": "validation-9", "predictions": b"1", "planned": False}
    with pytest.raises(ExperimentError) as pending:
        context.evaluate(components, **request)
    evaluation = pending.value.details["evaluation_id"]
    assert pending.value.details["missing_aggregate_ids"] == ["mean"]
    assert context.history.get_evaluation(context.run_id, evaluation)["state"] == "running"

    writer = RunContext(context.store, context.run_id, evaluation_id=evaluation)
    reference = {"kind": "result", "run_id": context.run_id,
                 "evaluation_id": evaluation, "result_id": "sample"}
    writer.publish_aggregate("mean", 1, {}, False, [reference])
    report = {"complete": True, "aggregate_ids": ["mean"]}
    assert writer.finish_evaluation(report)["state"] == "succeeded"

    replay = context.evaluate_async(components, **request)
    assert replay.evaluation_id == evaluation
    assert replay.result(timeout=10) == {**report, "evaluation_id": evaluation}
    context.join_evaluations()


def test_successful_training_closes_an_unfinished_validation(store):
    resolved = config(f"{__name__}.EarlyReport")
    resolved["plan"] = ["train"]
    with store.transaction(write=True) as db:
        run = store.create_run(db, store.create_branch(db), resolved)
        store.stage(db, run, "train", "running")
    context = RunContext(store, run)
    with pytest.raises(ExperimentError) as pending:
        context.evaluate(Components(resolved, Path.cwd()), predictions=b"1", planned=False)
    evaluation = pending.value.details["evaluation_id"]
    with store.transaction(write=True) as db:
        store.stage(db, run, "train", "succeeded")
        store.finish(db, run, "succeeded")
    assert store.status(run)["status"] == "succeeded"
    summary = context.history.get_evaluation(run, evaluation)
    assert summary["state"] == "failed" and summary["complete"] is None
    assert summary["reason"] and summary["result_count"] == 1


def test_snapshot_failure_has_an_evaluation_identity(task):
    class UnreadableModel:
        def __init__(self):
            self.attempts = 0

        def state_dict(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("cannot capture live weights")
            return {"weight": 1}

    context, components = task
    model = UnreadableModel()
    options = {"planned": False, "progress": {"step": 3}, "request_id": "measure-3"}
    future = context.evaluate_async(components, model, **options)
    with pytest.raises(RuntimeError, match="cannot capture live weights"):
        future.result(timeout=10)
    evaluation = context.history.get_evaluation(context.run_id, future.evaluation_id)
    assert evaluation["state"] == "failed"
    assert evaluation["complete"] is None and evaluation["result_count"] == 0
    assert evaluation["target"]["progress"] == {"step": 3}
    retry = context.evaluate_async(components, model, **options)
    assert retry.evaluation_id == future.evaluation_id
    with pytest.raises(ExperimentError, match="cannot capture live weights"):
        retry.result(timeout=10)
    assert model.attempts == 1
    context.join_evaluations()


def test_standalone_derived_evaluation_registers_its_sources(store, task):
    previous, _ = task
    source_run = previous.run_id
    source_eval, _ = previous.history.begin_evaluation(source_run, "original", {})
    previous.history.record_result(source_run, source_eval, "row", "input", 6)
    previous.history.finish_evaluation(source_run, source_eval, "succeeded", complete=True)

    directory = MOCK / "work"
    directory.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=directory, prefix="eval-lineage-") as work:
        source = Path(work)
        (source / "src").mkdir()
        (source / "src" / "derived.py").write_text(
            """from automl.evaluator import Evaluator

class Mean(Evaluator):
    uses_saved_measurements = True

    def __init__(self, origin):
        self.origin = origin

    @classmethod
    def evaluation_lineage(cls, settings):
        return {"relation": "reaggregation", "source_evaluations": [settings["origin"]]}

    def evaluate(self, components, context, model, *, predictions=None):
        run_id, evaluation_id = self.origin
        ref = {"kind": "result", "run_id": run_id, "evaluation_id": evaluation_id, "result_id": "row"}
        value = context.history.get_result(run_id, evaluation_id, "row")["value"]
        context.publish_aggregate("mean", value, {}, False, [ref])
        return {"complete": True, "aggregate_ids": ["mean"]}
"""
        )
        (source / "eval.yaml").write_text(yaml.safe_dump(config("derived.Mean", origin=[source_run, source_eval])))
        experiment = Experiment(store.root, source=source)
        run = experiment.start(experiment.branch(), "eval.yaml")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = experiment.status(run, reconcile=True)
            if status["status"] not in ("accepted", "running"):
                break
            time.sleep(0.05)
        assert status["status"] == "succeeded", status
        derived = experiment.list_derived_evaluations(source_run, source_eval, relation="reaggregation", limit=1)
        assert len(derived["evaluations"]) == 1
        evaluation = derived["evaluations"][0]
        assert evaluation["run_id"] == run
        summary = experiment.get_evaluation(run, evaluation["evaluation_id"])
        assert summary["source_evaluations"][0]["evaluation_id"] == source_eval
        assert summary["result_count"] == 0
        assert experiment.get_aggregate(run, evaluation["evaluation_id"], "mean")["value"] == 6
