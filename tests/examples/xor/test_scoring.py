import json
from types import SimpleNamespace

import numpy as np
import pytest

from automl.dataset import Sample
from examples.xor.components.scoring import Accuracy, ErrorRate


class Context:
    run_id = "run"
    evaluation_id = "eval"

    def __init__(self, samples):
        self.samples = samples
        self.results = {}
        self.aggregates = {}
        self.logs = []

    def open_dataset(self, components):
        return "fixed", SimpleNamespace(iter_samples=lambda *, split: iter(self.samples))

    def stop_requested(self):
        return False

    def log(self, kind, **detail):
        self.logs.append((kind, detail))

    def record_result(self, result_id, target, value):
        assert result_id not in self.results
        self.results[result_id] = (target, value)

    def publish_aggregate(self, aggregate_id, value, conditions, provisional, inputs):
        for reference in inputs:
            assert reference == {
                "run_id": "run", "evaluation_id": "eval", "kind": "result",
                "result_id": reference["result_id"],
            }
            assert reference["result_id"] in self.results
        self.aggregates[aggregate_id] = (value, provisional, [i["result_id"] for i in inputs])


def sample(i):
    return Sample(str(i), [1, -1], {"label": i % 2})


@pytest.mark.parametrize("minimum,complete", [(None, False), (95, True)])
def test_missing_prediction_uses_actual_denominator_and_attempts(minimum, complete):
    samples = [sample(i) for i in range(100, 196)]
    context = Context(samples)
    predictions = {s.sample_id: s.metadata["label"] for s in samples}
    del predictions["104"]
    predictions["other"] = 0
    result = Accuracy(on_error="retry", max_retries=7, min_count=minimum).evaluate(
        {}, context, None, predictions=json.dumps(predictions).encode()
    )
    assert result["planned_count"] == 96
    assert result["correct"] == result["count"] == 95
    assert result["failed_count"] == result["extra_prediction_count"] == 1
    assert result["excluded_count"] == 0
    assert result["failed"][0]["sample_id"] == "104"
    assert result["failed"][0]["reason"]
    assert result["failed"][0]["attempts"] == 1
    assert result["complete"] is complete
    assert result["min_count"] == (96 if minimum is None else 95)
    assert context.aggregates["accuracy"] == (1.0, not complete, result["sample_ids"])
    assert set(result["sample_ids"]) == set(predictions) - {"other"}


@pytest.mark.parametrize("content", [b"invalid", b'{"1": 0,"1": 1}', b"[]", b'{"extra":NaN}'])
def test_malformed_prediction_file_fails_before_scoring(content):
    context = Context([sample(1)])
    with pytest.raises(ValueError):
        Accuracy().evaluate({}, context, None, predictions=content)
    assert not context.results and not context.aggregates


def test_retry_model_but_propagate_result_storage_failure():
    class Model:
        calls = 0

        def predict(self, features):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient read failure")
            return np.array([1])

    context = Context([sample(0), sample(1)])
    model = Model()
    result = ErrorRate(on_error="retry", max_retries=2).evaluate({}, context, model)
    assert model.calls == 3
    assert result["complete"] and result["error_rate"] == 0.5
    assert context.results["0"][1]["attempts"] == 2
    assert context.results["1"][1]["attempts"] == 1
    assert context.aggregates["error_rate"] == (0.5, False, ["0", "1"])

    class BrokenStore(Context):
        def record_result(self, *args):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        Accuracy(on_error="retry").evaluate({}, BrokenStore([sample(1)]), model)
    assert model.calls == 4


def test_abort_and_empty_split_retain_their_completion_state():
    context = Context([sample(0), sample(1), sample(2)])
    result = Accuracy(on_error="abort").evaluate({}, context, None, predictions=b'{"0":1,"2":0}')
    assert result["sample_ids"] == ["0"]
    assert result["accuracy"] == 0.0 and result["provisional"]
    assert result["failed"][0]["sample_id"] == "1"
    assert result["excluded"][0]["sample_id"] == "2"
    assert result["extra_prediction_count"] == 0

    empty = Context([])
    result = ErrorRate().evaluate({}, empty, None, predictions=b"{}")
    assert result["error_rate"] is None and result["complete"] is False
    assert empty.aggregates["error_rate"] == (None, True, [])
