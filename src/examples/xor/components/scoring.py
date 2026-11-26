"""Binary classification metrics for the XOR experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from automl.evaluator import Evaluator
from automl.transform import Transform

from .features import require_components, sample_label, transform_features


def _read_predictions(content: bytes) -> dict[str, Any]:
    def unique_ids(pairs):
        result = {}
        for sample_id, value in pairs:
            if sample_id in result:
                raise ValueError(f"duplicate XOR prediction ID {sample_id!r}")
            result[sample_id] = value
        return result

    def invalid_number(number):
        raise ValueError(f"invalid JSON number {number}")

    values = json.loads(content, object_pairs_hook=unique_ids, parse_constant=invalid_number)
    if not isinstance(values, dict):
        raise ValueError("XOR predictions must map sample IDs to binary labels")
    return values


def _check_stop(context):
    if context.stop_requested():
        from automl.experiment.runtime import Interrupted

        raise Interrupted("XOR evaluation interrupted")


class Accuracy(Evaluator):
    metric = "accuracy"

    def __init__(self, split: str = "test", on_error: str = "abort", max_retries: int = 1,
                 min_count: int | None = None):
        self.split = split
        self.on_error = on_error
        self.max_retries = max_retries
        self.min_count = min_count

    @classmethod
    def selection_metrics(cls, settings):
        return (cls.metric,)

    @classmethod
    def resolve_settings(cls, settings: dict[str, Any], root: Path) -> dict[str, Any]:
        resolved = super().resolve_settings(settings, root)
        if not isinstance(resolved["split"], str) or not resolved["split"].strip():
            raise ValueError("split must be a non-empty name")
        if resolved["on_error"] not in ("abort", "continue", "retry"):
            raise ValueError("on_error must be abort, continue or retry")
        if type(resolved["max_retries"]) is not int or resolved["max_retries"] < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        minimum = resolved["min_count"]
        if minimum is not None and (type(minimum) is not int or minimum < 1):
            raise ValueError("min_count must be a positive integer")
        return resolved

    @classmethod
    def validate_components(cls, implementations):
        require_components(implementations, training=False)

    def _score(self, correct: int, count: int) -> float:
        return correct / count

    def evaluate(self, components, context, model, *, predictions=None):
        _check_stop(context)
        if (model is None) == (predictions is None):
            raise ValueError("XOR evaluation requires exactly one model or prediction file")

        by_id = None
        max_attempts = 1
        if predictions is not None:
            by_id = _read_predictions(predictions)

            def predict(sample):
                if sample.sample_id not in by_id:
                    raise ValueError(f"prediction file has no prediction for ID {sample.sample_id}")
                return by_id[sample.sample_id]
        else:
            transform = components.create("transform", Transform) if "transform" in components else None

            def predict(sample):
                features = transform_features(transform, [sample.data])
                return np.asarray(model.predict(features)).item()

            if self.on_error == "retry":
                max_attempts += self.max_retries

        version_id, dataset = context.open_dataset(components)
        context.log("dataset_split", component="dataset", version=version_id, split=self.split)
        seen = set()
        included = []
        failed = []
        excluded = []
        correct = 0

        for sample in dataset.iter_samples(split=self.split):
            _check_stop(context)
            sample_id = sample.sample_id
            if sample_id in seen:
                raise ValueError(f"duplicate sample ID {sample_id!r} in XOR split {self.split!r}")
            seen.add(sample_id)
            if failed and self.on_error == "abort":
                excluded.append({"sample_id": sample_id, "reason": "evaluation stopped after an error"})
                continue

            for attempt in range(1, max_attempts + 1):
                try:
                    label = sample_label(sample)
                    predicted = predict(sample)
                    if not isinstance(predicted, (int, float)) or predicted not in (0, 1):
                        raise ValueError("expected a binary prediction")
                except Exception as exc:
                    if attempt == max_attempts:
                        failure = {"sample_id": sample_id, "reason": str(exc), "attempts": attempt}
                        failed.append(failure)
                        context.log("evaluation_sample_failed", **failure)
                        break
                    _check_stop(context)
                else:
                    matched = bool(predicted == label)
                    target = {"sample_id": sample_id, "dataset_version": version_id, "split": self.split}
                    context.record_result(sample_id, target, {
                        "label": int(label), "prediction": int(predicted),
                        "correct": matched, "attempts": attempt,
                    })
                    included.append(sample_id)
                    correct += matched
                    break

        required = len(seen) if self.min_count is None else self.min_count
        count = len(included)
        complete = count > 0 and count >= required
        value = self._score(correct, count) if count else None
        conditions = {
            "metric": self.metric, "split": self.split, "dataset_version": version_id,
            "min_count": required, "on_error": self.on_error, "max_retries": self.max_retries,
        }
        inputs = [
            {"kind": "result", "run_id": context.run_id, "evaluation_id": context.evaluation_id,
             "result_id": sample_id}
            for sample_id in included
        ]
        _check_stop(context)
        context.publish_aggregate(self.metric, value, conditions, not complete, inputs)
        context.log("evaluation_summary", metric=self.metric, count=count, complete=complete)
        return {
            **conditions, self.metric: value, "planned_count": len(seen), "count": count,
            "correct": correct, "sample_ids": included, "failed": failed, "failed_count": len(failed),
            "excluded": excluded, "excluded_count": len(excluded), "complete": complete,
            "provisional": not complete, "aggregate_ids": [self.metric],
            "extra_prediction_count": len(by_id.keys() - seen) if by_id is not None else 0,
            "incomplete_reason": None if complete else
                f"{self.split}: required {max(1, required)} evaluated samples, got {count}",
        }


class ErrorRate(Accuracy):
    metric = "error_rate"

    def _score(self, correct: int, count: int) -> float:
        return (count - correct) / count
