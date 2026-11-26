"""Training interface for methods selected by experiment configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from automl.contracts import Component, State

if TYPE_CHECKING:
    from automl.experiment import Components, RunContext


class Trainer(Component, ABC):
    supports_resume: ClassVar[bool] = False

    def validate_resume(self, training_state: State, progress: dict[str, Any]) -> None:
        """Check the task-specific resume contract before admitting a new run."""
        if not self.supports_resume:
            raise ValueError(f"{type(self).__name__} does not support resuming training")

    @abstractmethod
    def train(
        self,
        components: Components,
        context: RunContext,
        *,
        start_method: Literal["new", "resume"],
    ) -> dict[str, Any]:
        """Train and report the outcome, including any checkpoint for evaluation.

        Record the initialized (or fully restored) state with
        `context.start_training` before the first update. Report the actual
        data used and poll for stop requests at safe points. On success return
        `usage_range` with the last used position per process (or an integer
        for a single process). Include `checkpoint` when selecting a saved
        model for post-training evaluation.
        """
