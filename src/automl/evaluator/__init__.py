"""Public contract for task-defined evaluation procedures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from automl.contracts import Component

if TYPE_CHECKING:
    from automl.experiment import Components, RunContext
    from automl.model import Model


class Evaluator(Component, ABC):
    """Measure a fixed model state or saved predictions under task-defined conditions."""

    # Opt in when evaluation can derive its results from existing measurements.
    uses_saved_measurements = False

    @classmethod
    def selection_metrics(cls, settings: dict[str, Any]) -> tuple[str, ...]:
        """Metrics available to a Trainer under these resolved settings."""
        return ()

    @classmethod
    def evaluation_lineage(cls, settings: dict[str, Any]) -> dict[str, Any]:
        """Identify source evaluations when starting a derived evaluation.

        Return ``relation`` and ``source_evaluations`` (run/evaluation ID pairs)
        for a continuation, correction, or reaggregation. Include ``reason`` for
        a correction. The worker registers these with the new evaluation ID
        before measuring or publishing aggregates.
        """
        return {}

    @abstractmethod
    def evaluate(
        self,
        components: Components,
        context: RunContext,
        model: Model | Mapping[str, Model] | None,
        *,
        predictions: bytes | None = None,
    ) -> dict[str, Any]:
        """Report measurements through context as they become final.

        For a new measurement, the caller supplies a model, named models, or a
        saved prediction artifact's bytes. A task aggregating existing results
        can use those measurements through its context. Publish aggregates
        with their inputs before returning. Include the boolean ``complete`` in
        the result to report whether the task's completion criterion was met.
        """


def evaluate(
    evaluator: Evaluator,
    components: Components,
    context: RunContext,
    model: Model | Mapping[str, Model] | None = None,
    *,
    predictions: bytes | None = None,
) -> dict[str, Any]:
    """Run the selected procedure on one evaluation target and validate its report."""
    if model is not None and predictions is not None:
        raise ValueError("model and predictions cannot be supplied together")
    if model is None and predictions is None and not evaluator.uses_saved_measurements:
        raise ValueError("evaluator requires a model or saved predictions")
    result = evaluator.evaluate(components, context, model, predictions=predictions)
    if not isinstance(result, dict) or not isinstance(result.get("complete"), bool):
        raise ValueError("evaluator must return a dict with a boolean 'complete' decision")
    return result
