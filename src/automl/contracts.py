"""Common construction and configuration hooks for experiment components."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Self


State = dict[str, Any] | bytes


class Component:
    @classmethod
    def resolve_settings(cls, settings: dict[str, Any], root: Path) -> dict[str, Any]:
        """Bind keyword settings and include constructor defaults before acceptance.

        Components can extend this hook to validate values or pin external inputs
        relative to the project root. It must also accept its own resolved output.
        """
        signature = inspect.signature(cls)
        bound = signature.bind(**settings)
        bound.apply_defaults()
        resolved = {}
        for name, parameter in signature.parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                resolved.update(bound.arguments[name])
            elif parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                resolved[name] = bound.arguments[name]
        return deepcopy(resolved)

    @classmethod
    def validate_components(cls, implementations: Mapping[str, type[Component]]) -> None:
        """Raise ValueError if another selected implementation is incompatible."""

    @classmethod
    def build(cls, settings: dict[str, Any], *, root: Path) -> Self:
        """Construct an instance from settings already resolved by Experiment."""
        return cls(**settings)
