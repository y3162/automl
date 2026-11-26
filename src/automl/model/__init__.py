"""Model construction and checkpoint state shared by all tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

from automl.contracts import Component, State


class Model(Component, ABC):
    initialization_report: dict[str, list[str]] | None = None

    @classmethod
    def build(
        cls,
        settings: dict[str, Any],
        *,
        root: Path,
        mode: Literal["new", "restore", "initialize"],
        state: State | None = None,
        retain: Sequence[str] | None = None,
    ) -> Self:
        """Build a fresh model, restore a checkpoint, or initialize a new branch.

        The model stays private until all requested state has been loaded.
        """
        if mode not in {"new", "restore", "initialize"}:
            raise ValueError(f"unknown model construction mode: {mode!r}")
        if (state is None) != (mode == "new"):
            raise ValueError(f"mode {mode!r} requires " + ("no checkpoint" if mode == "new" else "a checkpoint"))
        if retain is not None:
            if mode != "initialize":
                raise ValueError("retain is only valid for initialization from a checkpoint")
            if isinstance(retain, (str, bytes)) or not isinstance(retain, Sequence):
                raise ValueError("retain must be a sequence of state names")
            if any(not isinstance(name, str) or not name for name in retain):
                raise ValueError("each retain entry must be a nonempty state name")

        model = super().build(settings, root=root)
        if mode == "restore":
            model.load_state_dict(state)
        elif mode == "initialize":
            model.initialization_report = model.initialize_from(state, retain)
        return model

    @abstractmethod
    def state_dict(self) -> State:
        """Capture the complete model state, including learned non-parameter state."""

    @abstractmethod
    def load_state_dict(self, state: State) -> None:
        """Restore the complete state or raise with the missing or incompatible part."""

    def initialize_from(
        self, state: State, retain: Sequence[str] | None = None,
    ) -> dict[str, list[str]]:
        """Default to full inheritance. Implementations define partial-state names."""
        if retain is not None:
            raise ValueError(f"{type(self).__name__} cannot selectively retain {list(retain)!r}")
        self.load_state_dict(state)
        names = sorted(state) if isinstance(state, dict) else ["model"]
        return {"retained": names, "initialized": []}
