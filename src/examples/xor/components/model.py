"""Small NumPy classifiers sharing the XOR prediction and update contract."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from automl.model import Model

from .features import feature_array


class _BinaryClassifier(Model):
    parameters: dict[str, np.ndarray]

    def _forward(self, inputs):
        raise NotImplementedError

    def _gradients(self, inputs, hidden, delta):
        raise NotImplementedError

    def predict(self, features) -> np.ndarray:
        logits, _ = self._forward(feature_array(features))
        return (logits >= 0).astype(int)

    def update(self, features, labels, rate: float) -> float:
        inputs = feature_array(features)
        targets = np.asarray(labels)
        if not len(inputs) or targets.shape != (len(inputs),) or not np.isin(targets, (0, 1)).all():
            raise ValueError("expected one binary label for every example in a nonempty batch")
        logits, hidden = self._forward(inputs)
        probabilities = np.exp(-np.logaddexp(0.0, -logits))
        delta = (probabilities - targets) / len(inputs)
        for name, gradient in self._gradients(inputs, hidden, delta).items():
            self.parameters[name] -= rate * gradient
        return float(np.logaddexp(0.0, np.where(targets, -logits, logits)).mean())

    def state_dict(self) -> dict:
        return {name: array.tolist() for name, array in self.parameters.items()}

    def _load(self, state, names):
        replacements = {}
        for name in names:
            if name not in self.parameters or name not in state:
                raise ValueError(f"unknown or missing parameter {name!r}")
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != self.parameters[name].shape:
                raise ValueError(f"{name}: expected shape {self.parameters[name].shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite value in {name}")
            replacements[name] = value.copy()
        self.parameters.update(replacements)

    def load_state_dict(self, state) -> None:
        if not isinstance(state, dict) or state.keys() != self.parameters.keys():
            raise ValueError(f"checkpoint must contain exactly {sorted(self.parameters)}")
        self._load(state, self.parameters)

    def initialize_from(self, state, retain: Sequence[str] | None = None) -> dict:
        if retain is None:
            self.load_state_dict(state)
            retain = list(self.parameters)
        else:
            self._load(state, retain)
        return {"retained": sorted(retain), "initialized": sorted(self.parameters.keys() - set(retain))}


class TinyMLP(_BinaryClassifier):
    """Two inputs, one tanh hidden layer, and a logistic output."""

    def __init__(self, width: int = 8, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "hidden_weight": rng.normal(0.0, 0.7, (2, width)),
            "hidden_bias": rng.normal(0.0, 0.15, width),
            "output_weight": rng.normal(0.0, 0.7, width),
            "output_bias": np.array(0.0),
        }

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if type(values["width"]) is not int or values["width"] < 1:
            raise ValueError("width must be a positive integer")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        return values

    def _forward(self, inputs):
        p = self.parameters
        hidden = np.tanh(inputs @ p["hidden_weight"] + p["hidden_bias"])
        return hidden @ p["output_weight"] + p["output_bias"], hidden

    def _gradients(self, inputs, hidden, delta):
        hidden_delta = delta[:, None] * self.parameters["output_weight"] * (1 - hidden**2)
        return {
            "output_weight": hidden.T @ delta,
            "output_bias": delta.sum(),
            "hidden_weight": inputs.T @ hidden_delta,
            "hidden_bias": hidden_delta.sum(axis=0),
        }


class ProductLogistic(_BinaryClassifier):
    """Logistic regression on the interaction x1 * x2."""

    def __init__(self, weight: float = 0.0, bias: float = 0.0):
        self.parameters = {"weight": np.array(float(weight)), "bias": np.array(float(bias))}

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        for name, value in values.items():
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return values

    def _forward(self, inputs):
        product = inputs[:, 0] * inputs[:, 1]
        return product * self.parameters["weight"] + self.parameters["bias"], product

    def _gradients(self, inputs, product, delta):
        return {"weight": product @ delta, "bias": delta.sum()}
