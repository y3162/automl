"""Transform one or more supplied values independently of their storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from automl.contracts import Component


@dataclass(frozen=True)
class TransformResult:
    """Output and the conditions to use when replaying this stage."""

    data: Any
    reproduction: Any


class Transform(Component, ABC):
    @abstractmethod
    def apply(self, data: Any, *, reproduction: Any = None) -> TransformResult:
        """Apply this stage, replaying the recorded choices when supplied."""


class Identity(Transform):
    """Pass through both payload and annotations. No execution choices are made."""

    def apply(self, data: Any, *, reproduction: Any = None) -> TransformResult:
        if reproduction is not None and reproduction != {}:
            raise ValueError("Identity expects an empty reproduction mapping")
        return TransformResult(data=data, reproduction={})
