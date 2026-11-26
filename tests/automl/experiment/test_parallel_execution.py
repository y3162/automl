"""A task may start planned evaluation while its trainer is still making updates."""

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from automl.experiment import worker
from automl.experiment.runtime import Interrupted
from automl.experiment.store import Store


class Weight:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"weight": self.value}


class ParallelTask:
    def __init__(self, train_gate, evaluate_gate, evaluation_entered, outcome):
        self.train_gate = train_gate
        self.evaluate_gate = evaluate_gate
        self.evaluation_entered = evaluation_entered
        self.outcome = outcome
        self.specs = {"evaluator": {"implementation": "parallel_task.Measure", "settings": {}}}
        self.versions = {}

    def create(self, name, contract, **kwargs):
        if name == "model":
            return Weight(kwargs["state"]["weight"])
        return self

    def train(self, components, context, *, start_method):
        model = Weight(300)
        context.start_training(model.state_dict(), {}, {"step": 300})
        context.evaluate_async(components, model, progress={"step": 300})
        model.value += 1
        assert self.train_gate.wait(20), "test did not release the trainer"
        if self.outcome == "train_failure":
            raise ValueError("trainer failed after updating to 301")
        if context.stop_requested():
            raise Interrupted("trainer stopped safely")
        return {"usage_range": 0}

    def evaluate(self, components, context, model, *, predictions=None):
        self.evaluation_entered.set()
        assert self.evaluate_gate.wait(20), "test did not release the evaluator"
        context.record_result("weight", {"step": 300}, model.value)
        if self.outcome == "eval_failure":
            raise ValueError("measurement failed")
        if context.stop_requested():
            raise Interrupted("evaluator stopped safely")
        return {"complete": True}


def _wait_stage(store, run, name, state):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = store.status(run)
        if record["stages"][name] == state:
            return record["stages"]
        time.sleep(0.02)
    pytest.fail(f"stage {name} never reached {state}: {record['stages']}")


@pytest.mark.parametrize("outcome", ["success", "train_failure", "eval_failure", "stop_both"])
def test_overlapping_stage_outcomes_and_fixed_model(monkeypatch, outcome):
    root = Path(__file__).resolve().parents[3] / "data" / "mock" / "store"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as directory:
        store = Store(directory)
        config = {"plan": ["train", "evaluate"], "components": {}}
        with store.transaction(write=True) as db:
            run = store.create_run(db, store.create_branch(db), config, mode="train")
            store.update_run(db, run, status="running")
        train_gate, eval_gate, entered = (threading.Event() for _ in range(3))
        task = ParallelTask(train_gate, eval_gate, entered, outcome)
        monkeypatch.setattr(worker, "Components", lambda *args, **kwargs: task)
        thread = threading.Thread(target=worker.execute, args=(store, run), kwargs={"source": root, "tree": root})
        thread.start()
        try:
            assert entered.wait(10)
            if outcome == "stop_both":
                with store.transaction(write=True) as db:
                    store.update_run(db, run, stop_requested=True)
            elif outcome == "eval_failure":
                eval_gate.set()
                assert _wait_stage(store, run, "evaluate", "failed")["train"] == "running"
            else:
                train_gate.set()
                expected = "failed" if outcome == "train_failure" else "succeeded"
                assert _wait_stage(store, run, "train", expected)["evaluate"] == "running"
        finally:
            train_gate.set()
            eval_gate.set()
            thread.join(10)
        assert not thread.is_alive()
        stages = store.status(run)["stages"]
        if outcome == "stop_both":
            assert stages == {"train": "interrupted", "evaluate": "interrupted"}
        else:
            assert stages["train"] == ("failed" if outcome == "train_failure" else "succeeded")
            assert stages["evaluate"] == ("failed" if outcome == "eval_failure" else "succeeded")
        measured = [e["payload"] for e in store.history(run)["events"] if e["kind"] == "evaluation_result"]
        assert [item["value"] for item in measured] == [300]
