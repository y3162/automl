import shutil

import pytest

from automl.experiment import Experiment
from automl.mlflow import sync


def test_export_retry_and_projection_rebuild(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    source = Experiment(tmp_path / "store")
    store = source.store
    config = {"plan": ["train"], "components": {"trainer": {
        "implementation": "examples.xor.components.learning.FullBatch", "settings": {"rate": 0.1},
    }}}
    with store.transaction(write=True) as db:
        branch = store.create_branch(db)
        run = store.create_run(db, branch, config)
        store.event(db, run, "training_progress", {"rounds": 10, "loss": 0.25})
        store.finish(db, run, "succeeded")

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/display.db")
    artifacts = (tmp_path / "artifacts").as_uri()
    # The tracker stores the metric but the exporter never receives the response.
    log_metric = client.log_metric
    def disconnect(*args, **kwargs):
        log_metric(*args, **kwargs)
        raise OSError("lost response")

    with monkeypatch.context() as patch:
        patch.setattr(client, "log_metric", disconnect)
        with pytest.raises(OSError, match="lost response"):
            sync(source, client, artifact_location=artifacts)
    experiment = client.get_experiment_by_name("AutoML")
    interrupted, = client.search_runs([experiment.experiment_id])
    assert interrupted.data.tags["automl.sync_state"] == "error"

    mapping = sync(source, client, artifact_location=artifacts)
    assert mapping[run] == interrupted.info.run_id
    assert sync(source, client, artifact_location=artifacts) == mapping
    point, = client.get_metric_history(mapping[run], "train.loss")
    assert (point.step, point.value) == (10, 0.25)
    recorded = client.get_run(mapping[run])
    assert recorded.data.tags["automl.project_id"] == store.project_id
    assert recorded.data.tags["automl.sync_state"] == "synced"
    assert recorded.info.status == "FINISHED"

    copy = tmp_path / "restored" / "store"
    shutil.copytree(store.root, copy)
    restored = Experiment(copy)
    empty_client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path}/empty.db")
    rebuilt = sync(restored, empty_client, artifact_location=(tmp_path / "rebuilt-artifacts").as_uri())
    assert empty_client.get_run(rebuilt[run]).data.params == recorded.data.params
    point, = empty_client.get_metric_history(rebuilt[run], "train.loss")
    assert (point.step, point.value) == (10, 0.25)
