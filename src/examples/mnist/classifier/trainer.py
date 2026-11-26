"""Checkpoint mini-batch SGD at epoch boundaries with validation accuracy."""

import math
from itertools import islice

import numpy as np

from automl.dataset import Dataset
from automl.evaluator import Evaluator
from automl.experiment.runtime import Interrupted
from automl.model import Model
from automl.trainer import Trainer
from automl.transform import Transform


class MiniBatchSGD(Trainer):
    supports_resume = True

    def __init__(self, epochs: int = 2, limit: int | None = 256, batch_size: int = 32,
                 learning_rate: float = 0.25, seed: int = 17, selection_metric: str = "accuracy"):
        self.epochs = epochs
        self.limit = limit
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.selection_metric = selection_metric

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        for key in ("epochs", "batch_size", "limit"):
            if key == "limit" and values[key] is None:
                continue
            if type(values[key]) is not int or values[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        rate = values["learning_rate"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if values["selection_metric"] != "accuracy":
            raise ValueError("the classifier selects the checkpoint with the highest validation accuracy")
        return values

    @classmethod
    def validate_components(cls, implementations):
        for name, contract in (("dataset", Dataset), ("model", Model),
                               ("transform", Transform), ("validation", Evaluator)):
            if name not in implementations or not issubclass(implementations[name], contract):
                raise ValueError(f"MNIST classification training requires {name} implementing {contract.__name__}")
        if not callable(getattr(implementations["model"], "update", None)):
            raise ValueError("MNIST classification training requires a model with update()")

    def validate_resume(self, training_state, progress):
        epoch = progress.get("epoch")
        if (type(epoch) is not int or not 0 <= epoch <= self.epochs
                or type(training_state) is not dict or type(training_state.get("pending_validation")) is not bool
                or (epoch == 0 and training_state["pending_validation"])):
            raise ValueError("invalid epoch or validation state in MNIST classifier checkpoint")

    def _train_epoch(self, context, model, transform, samples, version, epoch, position):
        # An epoch-specific seed reproduces the next shuffle from the checkpoint's epoch.
        order = np.random.default_rng([self.seed, epoch]).permutation(len(samples))
        total_loss = 0.0
        for offset in range(0, len(samples), self.batch_size):
            batch = [samples[index] for index in order[offset:offset + self.batch_size]]
            inputs = transform.apply([sample.data for sample in batch])
            labels = [sample.metadata["label"] for sample in batch]
            loss = model.update(inputs.data, labels, self.learning_rate)
            position += 1
            context.add(usage_id=str(position), position=position, batch_id=position,
                        dataset_version=version, split="train",
                        sample_ids=[sample.sample_id for sample in batch],
                        details={"epoch": epoch, "reproduction": inputs.reproduction})
            if not math.isfinite(loss):
                raise ValueError(f"non-finite MNIST training loss at epoch {epoch}")
            total_loss += loss * len(batch)
        context.log("training_progress", epoch=epoch, loss=total_loss / len(samples))
        return position

    def train(self, components, context, *, start_method):
        version, dataset = context.open_dataset(components)
        samples = list(islice(dataset.iter_samples(split="train"), self.limit))
        if not samples:
            raise ValueError("empty MNIST training split")
        model = context.prepare_model(components)
        transform = components.create("transform", Transform)
        epoch = 0
        pending = False
        if start_method == "resume":
            initial = context.initial_states()
            self.validate_resume(initial["training"], initial["progress"])
            epoch = initial["progress"]["epoch"]
            pending = initial["training"]["pending_validation"]
        position = 0
        latest = None

        def checkpoint(score=None, metric=None):
            return context.save_checkpoint(model_state=model.state_dict(), training_state={"pending_validation": pending},
                                           progress={"epoch": epoch}, usage_range=position, score=score, metric=metric)

        context.log("dataset_split", component="dataset", version=version, split="train", count=len(samples))
        context.start_training(model_state=model.state_dict(), training_state={"pending_validation": pending},
                               progress={"epoch": epoch})
        try:
            while epoch < self.epochs or latest is None:
                if context.stop_requested():
                    raise Interrupted("MNIST classifier stopped at an epoch boundary")
                if not pending and epoch < self.epochs:
                    epoch += 1
                    position = self._train_epoch(context, model, transform, samples, version, epoch, position)
                pending = True
                result = context.evaluate(components, model, component="validation", progress={"epoch": epoch})
                if not result["complete"]:
                    raise ValueError("MNIST checkpoint selection needs a complete validation result")
                pending = False
                latest = checkpoint(result[self.selection_metric], {
                    "name": self.selection_metric, "direction": "max", "evaluation_id": context.last_evaluation,
                })
        except Interrupted:
            if pending or latest is None:
                checkpoint()
            raise

        best, score = context.best_checkpoint(direction="max")
        return {"checkpoint": best, "selection": "best", "epochs": epoch,
                "best_validation_accuracy": score, "usage_range": position}
