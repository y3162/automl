"""Exercise transactions and verify provenance independently of worker timing."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

import pytest

from automl.experiment.errors import ExperimentError
from automl.experiment.store import Store


@pytest.fixture
def store():
    base = Path("data/mock/store")
    base.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=base) as directory:
        yield Store(directory)


def test_same_request_and_competing_admissions(store):
    with store.transaction(write=True) as db:
        branch = store.create_branch(db)
    config = {"plan": ["train"]}
    barrier = Barrier(3)

    def admit(request):
        barrier.wait()
        try:
            with store.transaction(write=True) as db:
                previous = store.request(db, request, "start", {"branch": branch})
                if previous:
                    return previous
                run = store.create_run(db, branch, config)
                store.receipt(db, request, "start", {"branch": branch}, run)
                return run
        except ExperimentError as exc:
            assert exc.code == "branch_busy"
            return exc.details["active_run_id"]

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(admit, ["same", "same", "different"]))
    assert len(set(results)) == 1
    run = results[0]
    with store.transaction(write=True) as db:
        assert store.active_run(db, branch) == run
        # A process that never began training leaves the branch reusable.
        store.finish(db, run, "failed", error={"code": "launch_failed", "message": "no process"})
        assert store.active_run(db, branch) is None
        assert store.last_training(db, branch) is None
        assert sum(store.request(db, req, "start", {"branch": branch}) == run
                   for req in ("same", "different")) == 1
    assert len(store.history()["runs"]) == 1


def test_verify_keeps_origin_after_delete_and_checks_copy_and_order(store):
    with store.transaction(write=True) as db:
        parent = store.create_run(db, store.create_branch(db), {"plan": ["train"]})
        cp = store.artifacts.save(db, parent, "checkpoint", {"model": b"weights"}, order=1)
        store.finish(db, parent, "interrupted")
        branch = store.create_branch(db, origin_run=parent, origin_checkpoint=cp)
        child = store.create_run(db, branch, {"plan": ["train"]}, initial_checkpoint=cp)
        copied = store.artifacts.save(db, child, "start_state",
                                      {"model": store.artifacts.path(db, cp, "model")})
        data = {"artifact_id": copied}
        store.event(db, child, "start_state_saved", data)
        store.stage(db, child, "train", "running")
        store.stage(db, child, "train", "succeeded")
        store.finish(db, child, "succeeded")
    assert store.artifacts.delete(cp)["result"] == "deleted"
    assert store.verify()["issues"] == []

    with store.transaction(write=True) as db:
        copy = store.artifacts.path(db, copied, "model")
        copy.write_bytes(b"wrong")
        # The copied bytes and the save-before-start record must each be checked.
        db.execute("DELETE FROM events WHERE run_id=? AND kind='start_state_saved'", (child,))
        store.event(db, child, "start_state_saved", data)
    issues = store.verify(child)["issues"]
    assert {issue["run_id"] for issue in issues} == {child}
    assert any(issue.get("artifact_id") == copied for issue in issues)
    assert any(issue.get("stage") == "train" for issue in issues)


def test_read_only_and_tracking_snapshots(store):
    with store.transaction(write=True) as db:
        run = store.create_run(db, store.create_branch(db), {"plan": ["evaluate"]})
    revision = store.tracking_revisions()[run]
    observer = Store(store.root, read_only=True)
    assert observer.project_id == store.project_id
    with pytest.raises(ExperimentError):
        with observer.transaction(write=True):
            pass
    with store.transaction(write=True) as db:
        store.event(db, run, "note", {"message": "recorded"})
    snapshot = observer.tracking_snapshot(run)
    assert snapshot["revision"] > revision
    assert snapshot["events"][-1]["payload"] == {"message": "recorded"}
    assert snapshot["runs"][0]["stages"] == {"evaluate": "pending"}


def test_verify_detects_missing_committed_use_alongside_damaged_run(store):
    from automl.dataset import usage

    with store.transaction(write=True) as db:
        first = store.create_run(db, store.create_branch(db), {"plan": ["train"]})
        second = store.create_run(db, store.create_branch(db), {"plan": ["evaluate"]})
        usage.add(db, first, usage_id="sample-1", position=1, sample_ids=["s"], dataset_version="v1")
        checkpoint = store.artifacts.save(db, first, "checkpoint", {"model": b"content"},
                                          order=1, details={"usage_range": 1})
        usage.commit(db, first, checkpoint, 1)
    assert not store.verify()["issues"]
    with store.transaction(write=True) as db:
        db.execute("DELETE FROM dataset_usage WHERE run_id=?", (first,))
        db.execute("UPDATE runs SET record='broken JSON' WHERE id=?", (second,))
    issues = store.verify()["issues"]
    assert {(issue["run_id"], issue["item"]) for issue in issues} == {
        (first, "usage_range"), (second, "run_record")
    }
    assert any(issue.get("artifact_id") == checkpoint for issue in issues)
