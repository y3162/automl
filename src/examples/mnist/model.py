"""Numeric checkpoint format shared by the NumPy MNIST models."""
from io import BytesIO

import numpy as np

from automl.model import Model


class ArrayModel(Model):
    parameters: dict[str, np.ndarray]

    def state_dict(self) -> bytes:
        stream = BytesIO()
        np.savez(stream, **self.parameters)
        return stream.getvalue()

    def load_state_dict(self, state) -> None:
        self.initialize_from(state)

    def initialize_from(self, state, retain=None):
        if not isinstance(state, bytes):
            raise ValueError("expected an NPZ checkpoint")
        with np.load(BytesIO(state), allow_pickle=False) as saved:
            names = self.parameters.keys() if retain is None else set(retain)
            if retain is None and set(saved.files) != names:
                raise ValueError(f"expected parameters {sorted(names)}, got {sorted(saved.files)}")
            replacements = {}
            for name in names:
                if name not in self.parameters or name not in saved:
                    raise ValueError(f"unknown or missing parameter {name!r}")
                value = saved[name]
                expected = self.parameters[name]
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(f"{name} requires shape {expected.shape} and dtype {expected.dtype}")
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite value in {name}")
                replacements[name] = value
        self.parameters.update(replacements)
        return {"retained": sorted(names), "initialized": sorted(self.parameters.keys() - names)}
