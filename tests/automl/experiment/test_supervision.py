"""Exercise abnormal exits with real processes and the production history store."""

import json
import os
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import time

import pytest

from automl.experiment import engine
from automl.experiment.store import Store


# A small worker with a barrier allows the test to terminate the monitor at a
# known instant. It still claims the run, records stages and exits via the real
# supervisor protocol and the same interpreter/environment as a training task.
_WORK = """
import os
import signal
import time

def execute(store, run_id, *, source, tree):
    with store.transaction(write=True) as db:
        store.stage(db, run_id, 'train', 'running')
    deadline = time.monotonic() + 30
    while not (source / 'finish').is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError('Timed out waiting for test barrier')
        time.sleep(0.02)
    print('x' * 20000 + 'final output', flush=True)
    if (source / 'exception').exists():
        raise ValueError('error from the training stage')
    with store.transaction(write=True) as db:
        store.stage(db, run_id, 'train', 'succeeded')
    if (source / 'fatal').is_file():
        os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.fixture
def local_job():
    project = Path(__file__).resolve().parents[3]
    mock = project / "data" / "mock"
    (mock / "store").mkdir(parents=True, exist_ok=True)
    (mock / "work").mkdir(parents=True, exist_ok=True)
    with (TemporaryDirectory(dir=mock / "store") as directory,
          TemporaryDirectory(dir=mock / "work") as workspace):
        store, tree = Store(directory), Path(workspace)
        implementation = tree / "src" / "automl"
        shutil.copytree(project / "src" / "automl", implementation,
                        ignore=shutil.ignore_patterns("__pycache__"))
        with (implementation / "experiment" / "worker.py").open("a") as worker:
            worker.write("\n" + _WORK)
        with store.transaction(write=True) as db:
            branch = store.create_branch(db)
            run = store.create_run(db, branch, {"plan": ["train"]}, **engine.launch_owner())
        try:
            yield store, tree, run
        finally:
            (tree / "finish").touch()


def wait_for(store, run, condition):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        observed = engine.status(store, run, reconcile=True)
        if condition(observed):
            return observed
        if observed["status"] in {"succeeded", "failed", "interrupted"}:
            pytest.fail(f"Unexpected final state: {observed}")
        time.sleep(0.03)
    pytest.fail(f"Timed out waiting for run {run}: {store.status(run)}")


@pytest.mark.parametrize("outcome", ["success", "fatal", "exception"])
def test_reconcile_after_monitor_loss(local_job, outcome):
    store, tree, run = local_job
    (tree / outcome).touch()
    engine.launch(store, run, source=tree, tree=tree)
    running = wait_for(store, run, lambda row: row["stages"]["train"] == "running")
    os.kill(running["monitor_pid"], signal.SIGKILL)
    (tree / "finish").touch()
    result = wait_for(store, run, lambda row: row["status"] in {"succeeded", "failed"})
    assert result["status"] == ("succeeded" if outcome == "success" else "failed")
    assert result["state_confirmed"] and result["ended_at"]
    assert result["stages"]["train"] == ("failed" if outcome == "exception" else "succeeded")
    with store.transaction() as db:
        assert store.active_run(db, result["branch"]) is None
        if outcome != "success":
            diagnostics = next(a for a in store.artifacts.list(db, run) if a["kind"] == "diagnostics")
            record = json.loads(store.artifacts.read(db, diagnostics["id"], "diagnostics.json"))
            assert len(record["stdout_tail"].encode()) == 16 * 1024
            assert record["stdout_tail"].endswith("final output\n")
            assert record["error"]["code"] == "execution_failed"
            if outcome == "exception":
                assert record["error"]["details"]["stage"] == "train"
                assert record["error"]["message"] == "error from the training stage"
                assert "ValueError: error from the training stage" in record["traceback_tail"]


def test_stop_before_launch_prevents_late_worker_claim(local_job):
    store, tree, run = local_job
    engine.stop(store, run)
    engine.launch(store, run, source=tree, tree=tree)
    result = engine.status(store, run, reconcile=True)
    assert result["status"] == "interrupted"
    assert result["stages"] == {"train": "interrupted"}
    assert not result["training_started"]
    assert "pid" not in result


def test_reconciliation_requires_confirmed_process_identity(local_job):
    store, _, run = local_job
    owner = engine.launch_owner()
    # This PID is still live, but belongs to a different process start.
    stale = {**owner["launcher_process"], "start": -1}
    with store.transaction(write=True) as db:
        store.update_run(db, run, execution_host="unreachable.example", launcher_process=stale)
    unknown = engine.status(store, run, reconcile=True)
    assert unknown["status"] == "accepted" and not unknown["state_confirmed"]
    with store.transaction(write=True) as db:
        store.update_run(db, run, execution_host=owner["execution_host"])
    reader = Store(store.root, read_only=True)
    observed = engine.status(reader, run, reconcile=True)
    assert observed["error"]["code"] == "launch_failed"
    assert store.status(run)["status"] == "accepted"
    repaired = engine.status(store, run, reconcile=True)
    assert repaired["status"] == store.status(run)["status"] == "failed"
