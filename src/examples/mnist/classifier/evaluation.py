"""Evaluate a restored classifier or a saved mapping of sample IDs to digits."""

import json
from itertools import islice

import numpy as np

from automl.dataset import Dataset
from automl.evaluator import Evaluator
from automl.experiment.runtime import Interrupted
from automl.model import Model
from automl.transform import Transform


def _prediction_map(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"duplicate prediction for MNIST sample {key!r}")
        if type(value) is not int or not 0 <= value < 10:
            raise ValueError(f"invalid digit prediction for MNIST sample {key!r}")
        values[key] = value
    return values


class Accuracy(Evaluator):
    def __init__(self, split: str = "test", limit: int | None = 64, save_predictions: bool = False):
        self.split = split
        self.limit = limit
        self.save_predictions = save_predictions

    @classmethod
    def selection_metrics(cls, settings):
        return ("accuracy",)

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if values["split"] not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        limit = values["limit"]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer or null")
        if type(values["save_predictions"]) is not bool:
            raise ValueError("save_predictions must be boolean")
        return values

    @classmethod
    def validate_components(cls, implementations):
        if "dataset" not in implementations or not issubclass(implementations["dataset"], Dataset):
            raise ValueError("MNIST accuracy requires a dataset component")
        if "model" in implementations:
            model = implementations["model"]
            if not issubclass(model, Model) or not callable(getattr(model, "predict", None)):
                raise ValueError("MNIST accuracy requires a model with predict()")
            if "transform" not in implementations or not issubclass(implementations["transform"], Transform):
                raise ValueError("MNIST classifier inference requires a transform")

    def evaluate(self, components, context, model, *, predictions=None):
        saved = None
        transform = None
        if predictions is not None:
            saved = json.loads(predictions, object_pairs_hook=_prediction_map)
            if not isinstance(saved, dict):
                raise ValueError("MNIST predictions must map sample IDs to digits")
        else:
            transform = components.create("transform", Transform)

        version, dataset = context.open_dataset(components)
        observed = {}
        correct = 0
        for sample in islice(dataset.iter_samples(split=self.split), self.limit):
            if context.stop_requested():
                raise Interrupted("MNIST classification evaluation stopped")
            sample_id = sample.sample_id
            if sample_id in observed:
                raise ValueError(f"duplicate sample ID {sample_id!r} in MNIST split {self.split!r}")
            if saved is None:
                features = transform.apply([sample.data]).data
                prediction = np.asarray(model.predict(features)).item()
            elif sample_id in saved:
                prediction = saved[sample_id]
            else:
                raise ValueError(f"no prediction for MNIST sample {sample_id!r}")
            if type(prediction) is not int or not 0 <= prediction < 10:
                raise ValueError(f"invalid digit prediction for MNIST sample {sample_id!r}")
            label = sample.metadata["label"]
            matched = prediction == label
            correct += matched
            context.record_result(sample_id, {"sample_id": sample_id, "dataset_version": version, "split": self.split},
                                  {"label": label, "prediction": prediction, "correct": matched})
            observed[sample_id] = prediction
        if not observed:
            raise ValueError(f"empty MNIST evaluation split {self.split!r}")

        accuracy = correct / len(observed)
        conditions = {"metric": "accuracy", "dataset_version": version, "split": self.split, "limit": self.limit}
        inputs = [{"kind": "result", "run_id": context.run_id, "evaluation_id": context.evaluation_id,
                   "result_id": sample_id} for sample_id in observed]
        context.publish_aggregate("accuracy", accuracy, conditions, False, inputs)
        result = {**conditions, "accuracy": accuracy, "correct": correct, "count": len(observed),
                  "aggregate_ids": ["accuracy"], "complete": True}
        if self.save_predictions and saved is None:
            result["prediction_artifact"] = context.save_artifact("predictions.json", json.dumps(observed).encode())
        return result
