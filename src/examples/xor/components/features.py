"""Input conventions shared by the XOR trainer and evaluator."""

from __future__ import annotations

import numpy as np

from automl.dataset import Dataset
from automl.model import Model
from automl.transform import Transform


def require_components(implementations, *, training: bool) -> None:
    dataset = implementations.get("dataset")
    if dataset is None or not issubclass(dataset, Dataset):
        raise ValueError("XOR requires a dataset component implementing Dataset")
    if "transform" in implementations and not issubclass(implementations["transform"], Transform):
        raise ValueError("XOR transform must implement Transform")
    model = implementations.get("model")
    if model is None and not training:
        return  # Evaluating external predictions only needs labels.
    if model is None or not issubclass(model, Model):
        raise ValueError("XOR training requires a model component implementing Model")
    required = ("predict", "update") if training else ("predict",)
    for name in required:
        if not callable(getattr(model, name, None)):
            raise ValueError(f"XOR model {model.__name__} must implement {name}")


def feature_array(rows) -> np.ndarray:
    features = np.asarray(rows, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 2 or not np.isfinite(features).all():
        raise ValueError("expected a batch of finite (x1, x2) pairs")
    return features


def transform_features(transform, rows) -> np.ndarray:
    values = rows if transform is None else transform.apply(rows).data
    features = feature_array(values)
    if len(features) != len(rows):
        raise ValueError("XOR transform changed the number of samples")
    return features


def sample_label(sample) -> int:
    label = sample.metadata.get("label")
    if label not in (0, 1):
        raise ValueError(f"sample {sample.sample_id!r} must have a binary label")
    return int(label)
