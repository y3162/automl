"""NumPy digit classifiers for flattened IDX images."""

import numpy as np

from automl.transform import Transform, TransformResult
from examples.mnist.model import ArrayModel


class FlattenImages(Transform):
    def apply(self, data, *, reproduction=None):
        if reproduction is not None and reproduction != {}:
            raise ValueError("FlattenImages expects an empty reproduction mapping")
        images = np.asarray(data, dtype=np.float32)
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("expected a batch of 1 x 28 x 28 MNIST images")
        return TransformResult(images.reshape(len(images), 784), {})


def _check_features(features):
    if features.ndim != 2 or features.shape[1] != 784 or not np.isfinite(features).all():
        raise ValueError("expected a batch of 784 finite pixel values per image")


def _softmax_loss_gradient(logits, labels):
    count = len(logits)
    labels = np.asarray(labels)
    if (not count or labels.shape != (count,) or labels.dtype.kind not in "iu"
            or (labels < 0).any() or (labels >= 10).any()):
        raise ValueError("expected one digit label per training image")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    totals = exponentials.sum(axis=1, keepdims=True)
    rows = np.arange(count)
    loss = np.mean(np.log(totals[:, 0]) - shifted[rows, labels])
    delta = exponentials / totals
    delta[rows, labels] -= 1
    delta /= count
    return float(loss), delta


class SoftmaxClassifier(ArrayModel):
    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "weight": rng.normal(0, 0.01, (784, 10)).astype(np.float32),
            "bias": np.zeros(10, dtype=np.float32),
        }

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        return values

    def _logits(self, features):
        _check_features(features)
        return features @ self.parameters["weight"] + self.parameters["bias"]

    def predict(self, features):
        return self._logits(features).argmax(axis=1)

    def update(self, features, labels, learning_rate):
        loss, delta = _softmax_loss_gradient(self._logits(features), labels)
        self.parameters["weight"] -= learning_rate * (features.T @ delta)
        self.parameters["bias"] -= learning_rate * delta.sum(axis=0)
        return loss


class MLPClassifier(ArrayModel):
    """A fully connected ReLU hidden layer followed by a ten-class softmax."""

    def __init__(self, width: int = 128, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.parameters = {
            "hidden_weight": rng.normal(0, (2 / 784) ** 0.5, (784, width)).astype(np.float32),
            "hidden_bias": np.zeros(width, dtype=np.float32),
            "output_weight": rng.normal(0, width ** -0.5, (width, 10)).astype(np.float32),
            "output_bias": np.zeros(10, dtype=np.float32),
        }

    @classmethod
    def resolve_settings(cls, settings, root):
        values = super().resolve_settings(settings, root)
        if type(values["width"]) is not int or values["width"] < 1:
            raise ValueError("width must be a positive integer")
        if type(values["seed"]) is not int or values["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        return values

    def _forward(self, features):
        _check_features(features)
        p = self.parameters
        hidden = np.maximum(features @ p["hidden_weight"] + p["hidden_bias"], 0)
        return hidden, hidden @ p["output_weight"] + p["output_bias"]

    def predict(self, features):
        _, logits = self._forward(features)
        return logits.argmax(axis=1)

    def update(self, features, labels, learning_rate):
        hidden, logits = self._forward(features)
        loss, delta = _softmax_loss_gradient(logits, labels)
        p = self.parameters
        hidden_delta = (delta @ p["output_weight"].T) * (hidden > 0)
        p["output_weight"] -= learning_rate * (hidden.T @ delta)
        p["output_bias"] -= learning_rate * delta.sum(axis=0)
        p["hidden_weight"] -= learning_rate * (features.T @ hidden_delta)
        p["hidden_bias"] -= learning_rate * hidden_delta.sum(axis=0)
        return loss
