"""Full-batch and shuffled mini-batch gradient descent for XOR.

A round is one pass over the training split. Checkpoints are taken at round
boundaries so the shuffle generator and round number fully describe the next
batch sequence. Usage positions are local to the run, even on resume.
"""

from __future__ import annotations

import math

import numpy as np

from automl.trainer import Trainer
from automl.transform import Transform

from .features import feature_array, require_components, sample_label

# These names can also be used together with the trainers in configuration files.
from .scoring import Accuracy, ErrorRate

__all__ = ["FullBatch", "MiniBatch", "Accuracy", "ErrorRate"]


class FullBatch(Trainer):
    supports_resume = True

    def __init__(self, rounds: int = 100, rate: float = 0.8, save_every: int = 25,
                 seed: int = 17, selection_metric: str = "accuracy", minimize: bool = False,
                 final_checkpoint: str = "latest"):
        self.rounds = rounds
        self.rate = rate
        self.save_every = save_every
        self.seed = seed
        self.selection_metric = selection_metric
        self.minimize = minimize
        self.final_checkpoint = final_checkpoint

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        for key in ("rounds", "save_every", "batch_size"):
            if key in values and (type(values[key]) is not int or values[key] < 1):
                raise ValueError(f"{key} must be a positive integer")
        if type(values["rate"]) not in (int, float) or not math.isfinite(values["rate"]) or values["rate"] <= 0:
            raise ValueError("rate must be finite and positive")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(values["selection_metric"], str) or not values["selection_metric"].strip():
            raise ValueError("selection_metric must be a nonempty metric name")
        if type(values["minimize"]) is not bool:
            raise ValueError("minimize must be boolean")
        if values["final_checkpoint"] not in ("latest", "best"):
            raise ValueError("final_checkpoint must be latest or best")
        return values

    @classmethod
    def validate_components(cls, implementations):
        require_components(implementations, training=True)

    def batches(self, size: int, rng):
        yield np.arange(size)

    def validate_resume(self, training_state, progress):
        if not isinstance(training_state, dict) or not isinstance(progress, dict):
            raise ValueError("XOR checkpoint requires training state and progress mappings")
        completed = training_state.get("rounds")
        if type(completed) is not int or not 0 <= completed <= self.rounds:
            raise ValueError("invalid completed round count in XOR training checkpoint")
        if type(progress.get("rounds")) is not int or progress["rounds"] != completed:
            raise ValueError("checkpoint progress does not match the completed round count")
        try:
            np.random.default_rng(self.seed).bit_generator.state = training_state["rng"]
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid XOR shuffle generator state in checkpoint") from exc

    def _train_round(self, context, model, transform, version, features, labels, ids, rng,
                     round_number, position):
        total_loss = 0.0
        for batch in self.batches(len(ids), rng):
            inputs = features[batch]
            details = {"round": round_number}
            sample_ids = [ids[i] for i in batch]
            if transform is not None:
                applied = transform.apply(inputs)
                inputs = feature_array(applied.data)
                if len(inputs) != len(batch):
                    raise ValueError(f"XOR transform changed the number of samples in batch {sample_ids}")
                details["reproduction"] = applied.reproduction
            loss = model.update(inputs, labels[batch], self.rate)
            position += 1
            context.add(usage_id=str(position), position=position, sample_ids=sample_ids,
                        dataset_version=version, split="train", batch_id=str(position), details=details)
            if not math.isfinite(loss):
                raise ValueError(f"non-finite XOR training loss at round {round_number}")
            total_loss += loss * len(batch)
        return position, total_loss / len(ids)

    def _validate(self, components, context, model, completed):
        if "validation" not in components:
            return None, None
        result = context.evaluate(components, model, component="validation", progress={"rounds": completed})
        if not result["complete"]:
            return None, None
        score = result[self.selection_metric]
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"invalid validation score for {self.selection_metric!r}")
        metric = {"name": self.selection_metric, "direction": "min" if self.minimize else "max",
                  "evaluation_id": context.last_evaluation}
        return score, metric

    def train(self, components, context, *, start_method):
        from automl.experiment.runtime import Interrupted

        version, dataset = context.open_dataset(components)
        samples = list(dataset.iter_samples(split="train"))
        if not samples:
            raise ValueError("XOR training split is empty")
        ids = [sample.sample_id for sample in samples]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sample ID in XOR training split")
        features = feature_array([sample.data for sample in samples])
        labels = np.array([sample_label(sample) for sample in samples])
        transform = components.create("transform", Transform) if "transform" in components else None
        model = context.prepare_model(components)
        rng = np.random.default_rng(self.seed)
        completed = 0
        if start_method == "resume":
            initial = context.initial_states()
            state = initial["training"]
            self.validate_resume(state, initial["progress"])
            completed = state["rounds"]
            rng.bit_generator.state = state["rng"]
        elif start_method != "new":
            raise ValueError(f"unknown XOR training start method {start_method!r}")

        def training_state():
            return {"rounds": completed, "rng": rng.bit_generator.state}

        position = 0
        score = metric = latest = None

        def checkpoint():
            return context.save_checkpoint(model_state=model.state_dict(), training_state=training_state(),
                                           progress={"rounds": completed}, order=completed, usage_range=position,
                                           score=score, metric=metric, reuse_identical=True)

        context.log("dataset_split", component="dataset", version=version, split="train")
        context.start_training(model_state=model.state_dict(), training_state=training_state(),
                               progress={"rounds": completed})
        try:
            while completed < self.rounds:
                if context.stop_requested():
                    raise Interrupted("XOR training stopped at a round boundary")
                position, loss = self._train_round(context, model, transform, version, features, labels,
                                                   ids, rng, completed + 1, position)
                completed += 1
                score = metric = None
                if completed % self.save_every == 0 or completed == self.rounds:
                    context.log("training_progress", rounds=completed, loss=loss)
                    score, metric = self._validate(components, context, model, completed)
                    latest = checkpoint()
            if latest is None:  # A completed training run can also be resumed.
                score, metric = self._validate(components, context, model, completed)
                latest = checkpoint()
        except Interrupted:
            # Training is safe to stop even if the new save fails. The store
            # records the failed attempt and retains the last usable checkpoint.
            try:
                checkpoint()
            except Exception as exc:
                context.log("interruption_checkpoint_failed", rounds=completed, reason=str(exc))
            raise

        selected = latest
        if self.final_checkpoint == "best":
            best = context.best_checkpoint(direction="min" if self.minimize else "max")
            selected = best[0] if best else None
        result = {"checkpoint": selected, "selection": self.final_checkpoint, "rounds": completed,
                  "usage_range": position, "reason": "round_limit"}
        if selected is None:
            result.update(evaluation_disposition="skipped", evaluation_reason="no complete validation result")
        return result


class MiniBatch(FullBatch):
    def __init__(self, rounds: int = 20, rate: float = 0.35, save_every: int = 5, seed: int = 17,
                 batch_size: int = 32, selection_metric: str = "accuracy", minimize: bool = False,
                 final_checkpoint: str = "latest"):
        super().__init__(rounds, rate, save_every, seed, selection_metric, minimize, final_checkpoint)
        self.batch_size = batch_size

    def batches(self, size: int, rng):
        order = rng.permutation(size)
        for offset in range(0, size, self.batch_size):
            yield order[offset:offset + self.batch_size]
