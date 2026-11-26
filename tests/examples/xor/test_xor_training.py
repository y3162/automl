import numpy as np
import pytest

from automl.experiment.runtime import Interrupted
from examples.xor.components.data import InlineVersion
from examples.xor.components.learning import MiniBatch
from examples.xor.components.model import TinyMLP


def test_mlp_learning_freezes_checkpoints_and_respects_partial_initialization(tmp_path):
    features = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    labels = np.array([0, 1, 1, 0])
    model = TinyMLP()
    initial = model.state_dict()
    for _ in range(100):
        model.update(features, labels, 0.8)
    assert model.predict(features).tolist() == labels.tolist()
    trained = model.state_dict()
    assert trained != initial
    restored = TinyMLP.build({}, root=tmp_path, mode="restore", state=trained)
    assert restored.state_dict() == trained
    child = TinyMLP.build({}, root=tmp_path, mode="initialize", state=trained, retain=["hidden_weight"])
    assert child.state_dict() == {**initial, "hidden_weight": trained["hidden_weight"]}
    assert child.initialization_report["retained"] == ["hidden_weight"]
    with pytest.raises(ValueError, match="hidden_weight"):
        TinyMLP.build({"width": 4}, root=tmp_path, mode="initialize", state=trained, retain=["hidden_weight"])


class RoundBoundaryContext:
    """Capture exactly the state and batch IDs passed to the persistence API."""

    def __init__(self, initial=None, stop_at_round=None):
        self.initial = initial
        self.stop_at_round = stop_at_round
        self.latest = None
        self.batches = []
        self.version = InlineVersion({"train": [[1, 1, 0], [1, -1, 1], [-1, 1, 1], [-1, -1, 0]]})

    def open_dataset(self, components):
        return "v", self.version

    def prepare_model(self, components):
        model = TinyMLP()
        if self.initial:
            model.load_state_dict(self.initial["model_state"])
        return model

    def initial_states(self):
        return {"training": self.initial["training_state"], "progress": self.initial["progress"]}

    def log(self, kind, **payload):
        pass

    def start_training(self, **snapshot):
        expected = self.initial["progress"]["rounds"] if self.initial else 0
        assert snapshot["progress"]["rounds"] == expected

    def save_checkpoint(self, **snapshot):
        self.latest = snapshot
        return str(snapshot["order"])

    def add(self, **record):
        assert record["position"] == len(self.batches) + 1
        self.batches.append(record["sample_ids"])

    def stop_requested(self):
        return self.stop_at_round is not None and self.latest and self.latest["order"] == self.stop_at_round


def test_resumed_minibatches_match_uninterrupted_order_rng_and_weights():
    trainer = MiniBatch(rounds=6, save_every=2, seed=7, batch_size=2)
    uninterrupted = RoundBoundaryContext()
    trainer.train({}, uninterrupted, start_method="new")
    stopped = RoundBoundaryContext(stop_at_round=2)
    with pytest.raises(Interrupted):
        trainer.train({}, stopped, start_method="new")
    resumed = RoundBoundaryContext(initial=stopped.latest)
    report = trainer.train({}, resumed, start_method="resume")
    assert report["usage_range"] == 8
    assert stopped.batches + resumed.batches == uninterrupted.batches
    assert resumed.latest["training_state"] == uninterrupted.latest["training_state"]
    assert resumed.latest["model_state"] == uninterrupted.latest["model_state"]
    with pytest.raises(ValueError, match="does not match"):
        trainer.validate_resume(resumed.latest["training_state"], {"rounds": 0})
    with pytest.raises(ValueError, match="shuffle generator"):
        trainer.validate_resume({"rounds": 1, "rng": {}}, {"rounds": 1})
