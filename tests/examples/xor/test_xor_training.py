import numpy as np
import pytest

from automl.experiment.runtime import Interrupted
from examples.xor.components.data import InlineVersion
from examples.xor.components.learning import FullBatch, MiniBatch
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


class ValidationContext(RoundBoundaryContext):
    def __init__(self, scores, **kwargs):
        super().__init__(**kwargs)
        self.scores = iter(scores)
        self.validated = []

    def evaluate(self, components, model, *, component, progress):
        self.validated.append(progress["rounds"])
        self.last_evaluation = f"validation-{progress['rounds']}"
        score = next(self.scores)
        if isinstance(score, Interrupted):
            raise score
        return {"complete": score is not None, "accuracy": score, "error_rate": score}


@pytest.mark.parametrize("trainer_type,metric,minimize,target,miss", [
    (FullBatch, "accuracy", False, 1.0, 0.75),
    (MiniBatch, "error_rate", True, 0.0, 0.25),
])
def test_target_score_stops_at_a_complete_validation(trainer_type, metric, minimize, target, miss):
    trainer = trainer_type(rounds=8, save_every=2, selection_metric=metric,
                           minimize=minimize, target_score=target)
    context = ValidationContext([None, miss, target])
    result = trainer.train({"validation": True}, context, start_method="new")
    assert (result["rounds"], result["reason"], result["checkpoint"]) == (6, "target_score_reached", "6")
    assert context.validated == [2, 4, 6]
    assert context.latest["score"] == target
    assert context.latest["metric"]["evaluation_id"] == "validation-6"

    resumed = ValidationContext([target], initial=context.latest)
    result = trainer.train({"validation": True}, resumed, start_method="resume")
    assert (result["rounds"], result["reason"], result["usage_range"]) == (6, "target_score_reached", 0)
    assert resumed.batches == []
    assert resumed.latest["model_state"] == context.latest["model_state"]


def test_resume_interrupted_target_validation_before_updating():
    trainer = FullBatch(rounds=8, save_every=2, target_score=1.0)
    interrupted = ValidationContext([Interrupted("stop during validation")])
    with pytest.raises(Interrupted):
        trainer.train({"validation": True}, interrupted, start_method="new")
    resumed = ValidationContext([1.0], initial=interrupted.latest)
    result = trainer.train({"validation": True}, resumed, start_method="resume")
    assert resumed.validated == [2]
    assert result["reason"] == "target_score_reached"
    assert result["rounds"] == 2 and not resumed.batches


def test_resume_repeats_interrupted_validation_without_a_target_score():
    trainer = FullBatch(rounds=4, save_every=2)
    interrupted = ValidationContext([Interrupted("stop during validation")])
    with pytest.raises(Interrupted):
        trainer.train({"validation": True}, interrupted, start_method="new")
    resumed = ValidationContext([1.0, 0.75], initial=interrupted.latest)
    trainer.train({"validation": True}, resumed, start_method="resume")

    assert resumed.validated == [2, 4]
    assert len(resumed.batches) == 2
    assert resumed.latest["score"] == 0.75


@pytest.mark.parametrize("value", [True, "1", float("inf"), float("nan")])
def test_target_score_requires_a_finite_number(value, tmp_path):
    with pytest.raises(ValueError, match="target_score"):
        FullBatch.resolve_settings({"target_score": value}, tmp_path)
