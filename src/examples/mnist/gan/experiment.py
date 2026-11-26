"""Alternating SGD and held-out real/fake evaluation using two named AutoML models."""
import math
from itertools import islice

import numpy as np

from automl.dataset import Dataset
from automl.evaluator import Evaluator
from automl.experiment.runtime import Interrupted
from automl.trainer import Trainer

from .model import Discriminator, Generator


def _require_models(implementations):
    for name, contract in (("generator", Generator), ("discriminator", Discriminator), ("dataset", Dataset)):
        if name not in implementations or not issubclass(implementations[name], contract):
            raise ValueError(f"GAN requires {name} to implement {contract.__name__}")


class AdversarialTrainer(Trainer):
    """Each step samples a batch of training images and updates D, then G."""

    supports_resume = True

    def __init__(self, steps: int = 24, train_limit: int = 512, batch_size: int = 32,
                 rate: float = 0.05, seed: int = 3, save_every: int = 8):
        self.steps = steps
        self.train_limit = train_limit
        self.batch_size = batch_size
        self.rate = rate
        self.seed = seed
        self.save_every = save_every

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        for key in ("steps", "train_limit", "batch_size", "save_every"):
            if type(values[key]) is not int or values[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(values["rate"]) not in (int, float) or not math.isfinite(values["rate"]) or values["rate"] <= 0:
            raise ValueError("rate must be finite and positive")
        return values

    @classmethod
    def validate_components(cls, implementations):
        _require_models(implementations)
        if "validation" in implementations and not issubclass(implementations["validation"], Evaluator):
            raise ValueError("validation must implement Evaluator")

    def validate_resume(self, training_state, progress):
        if not isinstance(training_state, dict) or not isinstance(progress, dict):
            raise ValueError("GAN requires a training state and progress mapping")
        step = training_state.get("step")
        if type(step) is not int or not 0 <= step <= self.steps or progress != {"step": step}:
            raise ValueError("invalid step in GAN checkpoint")
        try:
            np.random.default_rng(self.seed).bit_generator.state = training_state["rng"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("invalid random generator state in GAN checkpoint") from exc

    def train(self, components, context, *, start_method):
        version, dataset = context.open_dataset(components)
        samples = list(islice(dataset.iter_samples(split="train"), self.train_limit))
        if not samples:
            raise ValueError("MNIST training split is empty")
        images = np.stack([s.data.reshape(784) for s in samples])
        generator = context.prepare_model(components, "generator")
        discriminator = context.prepare_model(components, "discriminator")
        rng = np.random.default_rng(self.seed)
        step = position = 0
        if start_method == "resume":
            state = context.initial_states()
            self.validate_resume(state["training"], state["progress"])
            step = state["progress"]["step"]
            rng.bit_generator.state = state["training"]["rng"]

        def states():
            return dict(model_states={"generator": generator.state_dict(), "discriminator": discriminator.state_dict()},
                        training_state={"step": step, "rng": rng.bit_generator.state}, progress={"step": step})

        context.start_training(**states())
        context.log("dataset_split", component="dataset", version=version, split="train")
        checkpoint = None
        while step < self.steps:
            if context.stop_requested():
                context.save_checkpoint(**states(), usage_range=position, reuse_identical=True)
                raise Interrupted("GAN stopped between updates")
            indices = rng.integers(len(samples), size=self.batch_size)
            noise_seed = int(rng.integers(2**63, dtype=np.uint64))
            noise = np.random.default_rng(noise_seed).standard_normal((self.batch_size, generator.latent_dim),
                                                                      dtype=np.float32)
            fake = generator.generate(noise)
            discriminator_loss = discriminator.update(images[indices], fake, self.rate)
            gradient, generator_loss = discriminator.generator_gradient(fake)
            generator.update(noise, fake, gradient, self.rate)
            step += 1
            position += 1
            context.add(usage_id=f"step-{step}", position=position, batch_id=step,
                        sample_ids=[samples[i].sample_id for i in indices], dataset_version=version,
                        split="train", details={"noise_seed": noise_seed})
            context.log("training_progress", step=step, discriminator_loss=discriminator_loss,
                        generator_loss=generator_loss)
            if step % self.save_every == 0 or step == self.steps:
                checkpoint = context.save_checkpoint(**states(), usage_range=position)
        if checkpoint is None:  # Resumed from the last update.
            checkpoint = context.save_checkpoint(**states(), usage_range=position)
        if "validation" in components:
            report = context.evaluate(components, checkpoint=checkpoint, component="validation")
            if not report["complete"]:
                raise ValueError(report["incomplete_reason"])
        return {"checkpoint": checkpoint, "usage_range": position}


def _sample_grid(images):
    """Encode the generated 28x28 images as a binary grayscale Netpbm sheet."""
    columns = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / columns)
    sheet = np.zeros((28 * rows, 28 * columns), dtype=np.uint8)
    for i, image in enumerate(images):
        row, col = divmod(i, columns)
        sheet[28*row:28*(row+1), 28*col:28*(col+1)] = np.rint(image.reshape(28, 28) * 255)
    return f"P5\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode("ascii") + sheet.tobytes()


class AdversarialEvaluation(Evaluator):
    """Measure balanced discriminator loss on real and seeded generated images."""

    def __init__(self, split: str = "test", count: int = 16, seed: int = 11):
        self.split = split
        self.count = count
        self.seed = seed

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if values["split"] not in ("validation", "test"):
            raise ValueError("split must be validation or test")
        if type(values["count"]) is not int or values["count"] <= 0:
            raise ValueError("count must be a positive integer")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        return values

    @classmethod
    def validate_components(cls, implementations):
        _require_models(implementations)

    def evaluate(self, components, context, model, *, predictions=None):
        if predictions is not None:
            raise ValueError("GAN evaluation requires a generator and discriminator checkpoint")
        generator, discriminator = model["generator"], model["discriminator"]
        version, dataset = context.open_dataset(components)
        rng = np.random.default_rng(self.seed)
        generated, losses, inputs = [], [], []
        for i, sample in enumerate(islice(dataset.iter_samples(split=self.split), self.count)):
            if context.stop_requested():
                raise Interrupted("GAN evaluation stopped between samples")
            noise = rng.standard_normal((1, generator.latent_dim), dtype=np.float32)
            fake = generator.generate(noise)
            real_logit = discriminator.logits(sample.data.reshape(1, 784)).item()
            fake_logit = discriminator.logits(fake).item()
            loss = float((np.logaddexp(0, -real_logit) + np.logaddexp(0, fake_logit)) / 2)
            context.record_result(sample.sample_id,
                                  {"sample_id": sample.sample_id, "dataset_version": version, "split": self.split,
                                   "generation_seed": self.seed, "generation_index": i},
                                  {"real_logit": real_logit, "generated_logit": fake_logit, "loss": loss})
            inputs.append({"kind": "result", "result_id": sample.sample_id, "run_id": context.run_id,
                           "evaluation_id": context.evaluation_id})
            generated.append(fake)
            losses.append(loss)
        complete = len(losses) == self.count
        mean_loss = float(np.mean(losses)) if losses else None
        conditions = {"split": self.split, "dataset_version": version, "seed": self.seed, "required": self.count}
        context.publish_aggregate("discriminator_loss", mean_loss, conditions, not complete, inputs)
        artifact = context.save_artifact("generated.pgm", _sample_grid(generated)) if generated else None
        return {"complete": complete, "count": len(losses), "discriminator_loss": mean_loss,
                "incomplete_reason": None if complete else f"required {self.count} samples, got {len(losses)}",
                "aggregate_ids": ["discriminator_loss"], "sample_grid": artifact}
