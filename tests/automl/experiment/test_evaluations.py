from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import pytest

from automl.experiment.errors import ExperimentError
from automl.experiment.evaluations import EvaluationHistory
from automl.experiment.store import Store


@pytest.fixture
def journal():
    mock = Path(__file__).resolve().parents[3] / "data/mock/store"
    mock.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=mock, prefix="evaluations-") as root:
        store = Store(Path(root) / "history")
        with store.transaction(write=True) as db:
            run = store.create_run(db, store.create_branch(db), {"plan": ["evaluate"]})
        yield EvaluationHistory(store), run, Path(root)


def test_concurrent_retry_and_evaluation_close(journal):
    history, run, _ = journal

    def acquire(db, eid):
        history.store.event(db, run, "lease_acquired", {"evaluation_id": eid})

    def begin(_):
        return history.begin_evaluation(run, {"step": 12}, {"version": 1},
                                        request_id="val-12", on_create=acquire)

    with ThreadPoolExecutor(max_workers=4) as pool:
        starts = list(pool.map(begin, range(4)))
    eid = starts[0][0]
    assert {start[0] for start in starts} == {eid}
    assert sum(created for _, created in starts) == 1
    with history.store.transaction() as db:
        assert db.execute("SELECT count(*) FROM events WHERE kind='lease_acquired'").fetchone()[0] == 1
    with pytest.raises(ExperimentError) as conflict:
        history.begin_evaluation(run, {"step": 13}, {"version": 1}, request_id="val-12")
    assert conflict.value.details["fields"] == ["target"]

    result = history.record_result(run, eid, "r1", {"sample_id": "s"}, 0.8)
    with pytest.raises(ExperimentError) as early:
        history.finish_evaluation(run, eid, "succeeded", complete=True, aggregate_ids=["mean"])
    assert early.value.details["missing_aggregate_ids"] == ["mean"]
    assert history.get_evaluation(run, eid)["complete"] is None

    ref = {"kind": "result", "run_id": run, "evaluation_id": eid, "result_id": "r1"}
    with pytest.raises(ExperimentError):
        history.publish_aggregate(run, eid, "mean", 0.8, {}, False, [ref, {**ref, "result_id": "r2"}])
    assert not history.list_aggregates(run, eid, limit=10)["aggregates"]
    aggregate = history.publish_aggregate(run, eid, "mean", 0.8, {}, False, [ref])
    end = history.finish_evaluation(run, eid, "succeeded", complete=True, aggregate_ids=["mean"])
    assert history.finish_evaluation(run, eid, "succeeded", complete=True, aggregate_ids=["mean"]) == end
    assert history.publish_aggregate(run, eid, "mean", 0.8, {}, False, [ref]) == aggregate
    assert history.record_result(run, eid, "r1", {"sample_id": "s"}, 0.8) == result
    assert history.get_evaluation(run, eid)["result_count"] == 1
    with pytest.raises(ExperimentError, match="closed"):
        history.record_result(run, eid, "r2", {}, 0.9)
    with pytest.raises(ExperimentError, match="Conflicting"):
        history.record_result(run, eid, "r1", {"sample_id": "s"}, 0.9)
    with pytest.raises(ExperimentError, match="Conflicting"):
        history.publish_aggregate(run, eid, "mean", 0.8, {}, False, [{**ref, "weight": 0.5}])
    with pytest.raises(ExperimentError, match="Conflicting"):
        history.finish_evaluation(run, eid, "interrupted", reason="stop")


def test_pages_keep_scope_state_and_boundary_after_copy(journal):
    history, run, root = journal
    first, _ = history.begin_evaluation(run, "C1", {})
    second, _ = history.begin_evaluation(run, "C2", {})
    for sample in ("x", "y", "z"):
        history.record_result(run, first, sample, sample, 0)
    cursor = history.read_results(run, first, limit=1)["next_cursor"]
    eval_cursor = history.list_evaluations(run, limit=1)["next_cursor"]
    history.record_result(run, first, "later", "more", 1)
    history.finish_evaluation(run, first, "succeeded", complete=True)
    history.finish_evaluation(run, second, "failed", complete=False, reason="no predictions")
    history.begin_evaluation(run, "C3", {})

    copytree(root / "history", root / "moved")
    restored = EvaluationHistory(Store(root / "moved", read_only=True))
    page = restored.read_results(run, first, cursor=cursor, limit=5)
    assert [item["result_id"] for item in page["results"]] == ["y", "z"]
    assert page["evaluation_state"] == "running" and page["next_cursor"] is None
    assert page == restored.read_results(run, first, cursor=cursor, limit=5)
    appended = restored.read_results(run, first, after="z", limit=5)
    assert [item["result_id"] for item in appended["results"]] == ["later"]
    assert appended["evaluation_state"] == "succeeded"
    evals = restored.list_evaluations(run, limit=5, cursor=eval_cursor)
    assert [(e["evaluation_id"], e["state"]) for e in evals["evaluations"]] == [(second, "running")]
    with pytest.raises(ExperimentError):
        restored.read_results(run, second, cursor=cursor, limit=2)
    with pytest.raises(ExperimentError):
        restored.read_results(run, first, cursor=cursor, after="y", limit=2)
    with pytest.raises(ExperimentError) as invalid:
        restored.read_results(run, first, cursor=cursor, limit=True)
    assert invalid.value.code == "invalid_request"


def test_cross_run_sources_and_direct_uses(journal):
    history, original_run, _ = journal
    old, _ = history.begin_evaluation(original_run, "C1", {})
    history.record_result(original_run, old, "previous", {"version": "v1", "sample": 2}, 3)
    base = {"kind": "result", "run_id": original_run, "evaluation_id": old, "result_id": "previous"}
    history.publish_aggregate(original_run, old, "old-metric", 3, {}, False, [base])
    with history.store.transaction(write=True) as db:
        new_run = history.store.create_run(db, history.store.create_branch(db), {"plan": ["evaluate"]})
    new, _ = history.begin_evaluation(
        new_run, "C1", {"version": "v2"}, relation="correction",
        source_evaluations=[(original_run, old)], reason="updated labels",
    )
    input_aggregate = {"kind": "aggregate", "run_id": original_run, "evaluation_id": old,
                       "aggregate_id": "old-metric"}
    weighted = [{**input_aggregate, "role": role} for role in ("baseline", "reference")]
    history.publish_aggregate(new_run, new, "compared", 3, {"unit": "score"}, False, weighted)
    decision = history.record_decision(new_run, new, "continue", aggregate_id="compared", request_id="d")
    assert decision == history.record_decision(new_run, new, "continue", aggregate_id="compared",
                                               request_id="d")
    assert history.get_evaluation(new_run, new)["result_count"] == 0
    assert history.get_evaluation(new_run, new)["source_evaluations"][0]["reason"] == "updated labels"
    derived = history.list_derived_evaluations(original_run, old, relation="correction", limit=5)
    assert [(e["run_id"], e["evaluation_id"]) for e in derived["evaluations"]] == [(new_run, new)]
    inputs = history.read_aggregate_inputs(new_run, new, "compared", limit=5)
    assert inputs["inputs"] == weighted
    uses = history.list_uses(original_run, old, aggregate_id="old-metric", limit=5)
    assert uses["uses"] == [{"kind": "aggregate", "run_id": new_run, "evaluation_id": new,
                             "aggregate_id": "compared"}]
    uses = history.list_uses(new_run, new, aggregate_id="compared", limit=5)
    assert uses["uses"] == [{"kind": "decision", "run_id": new_run, "evaluation_id": new,
                             "decision_id": decision}]
