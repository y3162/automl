"""Check classifier updates and the two paths for evaluating digit predictions."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from automl.experiment import Experiment
from examples.mnist.classifier.model import FlattenImages, MLPClassifier, SoftmaxClassifier


def test_softmax_learns_digit_labels_and_restores_parameters(tmp_path):
    images = np.zeros((10, 1, 28, 28), dtype=np.float32)
    labels = np.arange(10)
    images[labels, 0, 0, labels] = 1
    transform = FlattenImages()
    features = transform.apply(images)
    np.testing.assert_array_equal(transform.apply(images, reproduction=features.reproduction).data, features.data)
    model = SoftmaxClassifier(seed=3)
    loss = model.update(features.data, labels, 1.0)
    assert 0 < model.update(features.data, labels, 1.0) < loss
    saved = model.state_dict()
    restored = SoftmaxClassifier.build({"seed": 1}, root=tmp_path, mode="restore", state=saved)
    np.testing.assert_array_equal(restored.predict(features.data), labels)
    with pytest.raises(ValueError, match="digit label"):
        restored.update(features.data, labels + 1, 1.0)


def test_mlp_minibatch_gradients_match_cross_entropy(tmp_path):
    features = np.zeros((3, 784), dtype=np.float32)
    features[:, 0] = [1.0, -0.5, 0.75]
    labels = [1, 3, 4]
    probe = MLPClassifier(width=2)
    probe.parameters["hidden_weight"][0] = [1.0, -1.0]
    trained = MLPClassifier.build({"width": 2}, root=tmp_path, mode="restore", state=probe.state_dict())
    rate = 0.1
    trained.update(features, labels, rate)

    for name, index in (("hidden_weight", (0, 0)), ("hidden_bias", 1),
                        ("output_weight", (0, 1)), ("output_bias", 4)):
        before = probe.parameters[name][index].item()
        losses = []
        for offset in (-0.001, 0.001):
            probe.parameters[name][index] = before + offset
            _, logits = probe._forward(features)
            logits = logits.astype(np.float64)
            losses.append(np.mean(np.logaddexp.reduce(logits, axis=1) - logits[np.arange(3), labels]))
        probe.parameters[name][index] = before
        gradient = (before - trained.parameters[name][index].item()) / rate
        assert gradient == pytest.approx((losses[1] - losses[0]) / 0.002, abs=0.0001)


@pytest.mark.parametrize("train_config", ["train", "mlp"])
def test_classifier_scores_best_checkpoint_and_reuses_predictions(mnist_project, train_config):
    configs = Path("configs/examples/mnist/classifier")
    project = Path(__file__).resolve().parents[3]
    shutil.copytree(project / configs, mnist_project / configs)
    dataset = {"settings": {"validation_size": 4}}
    for name in ("train", "evaluate"):
        overrides = {"dataset": dataset}
        if name == "train":
            overrides["trainer"] = {"settings": {"limit": 16, "batch_size": 5, "epochs": 2}}
        parent = train_config if name == "train" else name
        config = {"extends": (configs / f"{parent}.yaml").as_posix(), "components": overrides}
        (mnist_project / f"{name}.yaml").write_text(yaml.safe_dump(config))

    experiment = Experiment(mnist_project / "data/mock/store", source=mnist_project)
    branch = experiment.branch()
    run = experiment.start(branch, "train.yaml")
    status = experiment.wait(run, timeout=60)
    assert status["stages"] == {"train": "succeeded", "evaluate": "succeeded"}, status
    usage = experiment.history(run)["usage"]
    batches = usage["batches"]
    assert [batch["batch_id"] for batch in batches] == list(range(1, 9))
    assert all(batch["complete"] for batch in batches)
    assert [row["batch_id"] for row in usage["records"]] == [batch["batch_id"] for batch in batches]
    assert [len(row["sample_ids"]) for row in usage["records"]] == [5, 5, 5, 1] * 2
    validations = experiment.list_evaluations(run, limit=3)["evaluations"]
    assert all(e["conditions"]["versions"] == status["dataset_versions"] for e in validations)
    first, second, test = validations
    assert first["result"]["count"] == second["result"]["count"] == 4
    assert test["result"]["count"] == 8
    assert test["target"]["metric"]["evaluation_id"] in {first["evaluation_id"], second["evaluation_id"]}
    scores = {item["evaluation_id"]: item["result"]["accuracy"] for item in (first, second)}
    assert scores[test["target"]["metric"]["evaluation_id"]] == max(scores.values())

    artifact = test["result"]["prediction_artifact"]
    rescored = experiment.start(branch, "evaluate.yaml", prediction_artifact=artifact)
    assert experiment.wait(rescored, timeout=60)["status"] == "succeeded"
    after, = experiment.list_evaluations(rescored, limit=1)["evaluations"]
    assert after["target"]["prediction_artifact"] == artifact
    assert after["result"]["accuracy"] == test["result"]["accuracy"]
    original = experiment.read_results(run, test["evaluation_id"], limit=8)["results"]
    repeated = experiment.read_results(rescored, after["evaluation_id"], limit=8)["results"]
    assert [(row["result_id"], row["value"]) for row in original] == [
        (row["result_id"], row["value"]) for row in repeated
    ]
    assert experiment.verify(run)["ready"] and experiment.verify(rescored)["ready"]
