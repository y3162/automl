"""Public API for local machine-learning experiments."""

from .api import Experiment
from .config import Components
from .errors import ExperimentError

__all__ = ["Components", "Experiment", "ExperimentError", "Interrupted", "RunContext"]


def __getattr__(name):
    if name in {"Interrupted", "RunContext"}:
        from . import runtime
        return getattr(runtime, name)
    raise AttributeError(name)
