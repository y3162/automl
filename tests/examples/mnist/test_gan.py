"""Exercise a two-model checkpoint and deterministic GAN sampling on CPU."""
import shutil
from pathlib import Path

import numpy as np
import yaml

from automl.experiment import Experiment
from examples.mnist.gan.model import Discriminator, Generator


def test_adversarial_updates_change_both_models():
    generator = Generator(latent_dim=3)
    discriminator = Discriminator()
    noise = np.random.default_rng(17).standard_normal((4, 3), dtype=np.float32)
    real = np.zeros((4, 784), dtype=np.float32)
    fake = generator.generate(noise)
    initial_logits = discriminator.logits(fake)
    loss = discriminator.update(real, fake, rate=0.001)
    assert np.isfinite(loss)
    assert np.all(discriminator.logits(fake) < initial_logits)

    gradient, loss = discriminator.generator_gradient(fake)
    generator.update(noise, fake, gradient, rate=0.01)
    updated = generator.generate(noise)
    assert np.all((0 <= updated) & (updated <= 1))
    assert discriminator.generator_gradient(updated)[1] < loss


def test_gan_resume_restores_both_models_and_noise_sequence(mnist_project):
    configs = Path("configs/examples/mnist/gan")
    repo = Path(__file__).resolve().parents[3]
    shutil.copytree(repo / configs, mnist_project / configs)
    config = {
        "extends": (configs / "train.yaml").as_posix(),
        "components": {
            "dataset": {"settings": {"validation_size": 4}},
            "trainer": {"settings": {"steps": 4, "save_every": 2, "batch_size": 3, "train_limit": 12}},
            "validation": {"settings": {"count": 3}},
            "evaluator": {"settings": {"count": 4}},
        },
    }
    (mnist_project / "train.yaml").write_text(yaml.safe_dump(config))
    experiment = Experiment(mnist_project / "data/mock/store", source=mnist_project)
    branch = experiment.branch()
    run = experiment.start(branch, "train.yaml")
    status = experiment.wait(run, timeout=60)
    assert status["stages"] == {"train": "succeeded", "evaluate": "succeeded"}, status
    history = experiment.history(run)
    progress = [event["payload"] for event in history["events"] if event["kind"] == "training_progress"]
    assert [entry["step"] for entry in progress] == [1, 2, 3, 4]
    assert all(np.isfinite([entry["discriminator_loss"], entry["generator_loss"]]).all() for entry in progress)
    first, last = [a for a in history["artifacts"] if a["kind"] == "checkpoint"]
    assert [a["details"]["progress"] for a in [first, last]] == [{"step": 2}, {"step": 4}]
    assert last["details"]["model_names"] == ["discriminator", "generator"]
    validation, test = experiment.list_evaluations(run, limit=2)["evaluations"]
    assert validation["result"]["count"] == 3 and test["result"]["count"] == 4
    assert test["target"]["checkpoint_id"] == last["id"]
    grid_id = test["result"]["sample_grid"]
    grid = next(a["files"]["generated.pgm"] for a in history["artifacts"] if a["id"] == grid_id)
    magic, dims, depth, pixels = (experiment.root / grid["path"]).read_bytes().split(b"\n", 3)
    assert (magic, dims, depth, len(pixels)) == (b"P5", b"56 56", b"255", 56 * 56)
    assert len(set(pixels)) > 1

    # Retain only the intermediate checkpoint in this isolated test store.
    # The next run must repeat the final two updates, including their random draws.
    experiment.delete_checkpoint(last["id"])
    resumed = experiment.resume(branch)
    status = experiment.wait(resumed, timeout=60)
    assert status["status"] == "succeeded", status
    assert status["resume_checkpoint"] == first["id"]
    replayed = experiment.history(resumed)
    checkpoint, = [a for a in replayed["artifacts"] if a["kind"] == "checkpoint"]
    assert checkpoint["details"]["usage_range"] == 2
    assert checkpoint["files"]["states"]["sha256"] == last["files"]["states"]["sha256"]
    _, retest = experiment.list_evaluations(resumed, limit=2)["evaluations"]
    assert retest["result"]["discriminator_loss"] == test["result"]["discriminator_loss"]
    regenerated = next(a for a in replayed["artifacts"] if a["id"] == retest["result"]["sample_grid"])
    assert regenerated["files"]["generated.pgm"]["sha256"] == grid["sha256"]
    assert experiment.verify(resumed)["ready"]
