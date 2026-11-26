from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from automl.experiment.errors import ExperimentError
from automl.experiment.store import Store


@pytest.fixture
def run_store():
    mock = Path(__file__).resolve().parents[3] / "data/mock/store"
    mock.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=mock, prefix="checkpoints-") as root:
        store = Store(Path(root))
        with store.transaction(write=True) as db:
            branch = store.create_branch(db)
            run = store.create_run(db, branch, {"plan": ["train"]})
        yield store, run


def checkpoint(store, run, step, wanted_id=None):
    with store.transaction(write=True) as db:
        return store.artifacts.save(db, run, "checkpoint",
                                    {"model": b"model", "trainer": b"trainer"},
                                    details={"progress": {"step": step}},
                                    order=step, artifact_id=wanted_id)


def test_latest_uses_training_order_and_skips_corrupt_files(run_store):
    store, run = run_store
    newer = checkpoint(store, run, 301)
    older = checkpoint(store, run, 300)
    with store.transaction() as db:
        assert store.artifacts.resolve(db, run)["id"] == newer
        path = store.artifacts.path(db, newer, "model")
    path.write_bytes(b"changed")
    with store.transaction() as db:
        assert store.artifacts.resolve(db, run + ":latest")["id"] == older
        with pytest.raises(ExperimentError) as error:
            store.artifacts.resolve(db, newer)
        assert error.value.details["artifact_id"] == newer
        report = store.artifacts.verify(db, run)
    assert report["checked"] == 4
    assert [(issue["run_id"], issue["artifact_id"]) for issue in report["issues"]] == [(run, newer)]


def test_unpublished_file_write_can_be_retried_with_same_id(run_store):
    store, run = run_store
    with pytest.raises(RuntimeError):
        with store.transaction(write=True) as db:
            store.artifacts.save(db, run, "checkpoint", {"model": b"uncommitted"},
                                 order=2, artifact_id="C2")
            raise RuntimeError("dataset usage could not be committed")
    with store.transaction() as db:
        assert not store.artifacts.list(db, run)
    assert not any((store.root / "runs" / run / "artifacts").iterdir())
    assert checkpoint(store, run, 2, "C2") == "C2"
    with store.transaction() as db:
        assert store.artifacts.read(db, "C2", "model") == b"model"


def test_deletion_waits_for_every_user_and_preserves_copies(run_store, monkeypatch):
    store, run = run_store
    saved = checkpoint(store, run, 400)
    files = store.artifacts
    with store.transaction(write=True) as db:
        files.acquire(db, saved, run + ":E1")
        files.acquire(db, saved, run + ":E2")
        copied = files.save(db, run, "start_state", {"model": files.path(db, saved, "model")})
        files.release(db, run + ":E1", saved)
    assert files.delete(saved) == {"checkpoint_id": saved, "result": "in_use"}
    with store.transaction(write=True) as db:
        assert files.get(db, saved)["checkpoint_state"] == "retained"
        files.release_run(db, run)
        trainer_file = files.path(db, saved, "trainer")
        model_file = files.path(db, saved, "model")

    unlink = Path.unlink

    def fail_during_unlink(path, *, missing_ok=False):
        if path == trainer_file:
            raise OSError("unavailable")
        return unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_during_unlink)
        with pytest.raises(ExperimentError):
            files.delete(saved)
    assert not model_file.exists() and trainer_file.is_file()
    with store.transaction() as db:
        with pytest.raises(ExperimentError) as error:
            files.acquire(db, saved, run + ":E3")
        assert error.value.details == {"checkpoint_id": saved, "checkpoint_state": "deleting"}
        assert files.verify(db, run)["pending_deletions"] == [saved]
        assert not files.verify(db)["issues"]

    assert files.delete(saved) == files.delete(saved) == {"checkpoint_id": saved, "result": "deleted"}
    with store.transaction() as db:
        assert files.get(db, saved)["details"]["progress"] == {"step": 400}
        assert files.read(db, copied, "model") == b"model"
        assert not files.verify(db)["issues"]


def test_verify_accumulates_invalid_indexes_and_missing_files(run_store):
    store, run = run_store
    missing = checkpoint(store, run, 1)
    empty = checkpoint(store, run, 2)
    undecodable = checkpoint(store, run, 3)
    with store.transaction(write=True) as db:
        store.artifacts.path(db, missing, "trainer").unlink()
        db.execute("UPDATE artifacts SET files='{}' WHERE id=?", (empty,))
        db.execute("UPDATE artifacts SET files='bad JSON' WHERE id=?", (undecodable,))
    with store.transaction() as db:
        issues = store.artifacts.verify(db)["issues"]
    assert {issue["artifact_id"] for issue in issues} == {missing, empty, undecodable}
