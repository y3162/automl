import sqlite3
from contextlib import closing

import pytest

from automl.dataset import usage
from automl.experiment.errors import ExperimentError


@pytest.fixture
def db():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
            CREATE TABLE runs (id TEXT PRIMARY KEY);
            CREATE TABLE artifacts (id TEXT PRIMARY KEY, run_id TEXT, kind TEXT);
            INSERT INTO runs VALUES ('run');
            INSERT INTO artifacts VALUES ('saved', 'run', 'checkpoint');
        """ + usage.USAGE_SCHEMA)
        yield connection


def record(db, position, process, samples, **kwargs):
    usage.add(db, "run", usage_id=f"{position}.{process}", position=position,
              process=process, sample_ids=samples, dataset_version="v1", split="train", **kwargs)


def test_partial_batch_retries_and_order(db):
    record(db, 1, 0, ["A", "B"], batch_id="first", processes=[1, 0])
    partial = usage.read_usage(db, "run", batch_id="first")
    assert partial["batches"] == [
        {"batch_id": "first", "processes": [0, 1], "complete": False, "missing_processes": [1]}
    ]
    with pytest.raises(ExperimentError) as err:
        usage.commit(db, "run", "saved", {0: 1, 1: 1})
    assert err.value.details["missing"] == [dict(process=1, start=1, end=1)]
    assert usage.read_usage(db, "run")["commits"] == []

    for _ in range(2):
        record(db, 1, 1, ["B", "C"], batch_id="first")
    with pytest.raises(ExperimentError) as err:
        record(db, 1, 1, ["C", "B"], batch_id="first")
    assert err.value.details["fields"] == ["sample_ids"]
    record(db, 2, 0, ["D"])
    usage.commit(db, "run", "saved", {"0": 1, "1": 1})
    usage.commit(db, "run", "saved", {1: 1, 0: 1})
    snapshot = usage.read_usage(db, "run", batch_id="first")
    assert [r["sample_ids"] for r in snapshot["records"]] == [["A", "B"], ["B", "C"]]
    assert snapshot["batches"][0]["complete"]
    assert snapshot["commits"] == [{"checkpoint_id": "saved", "usage_range": {"0": 1, "1": 1}}]
    process_view = usage.read_usage(db, "run", process=1)
    assert [r["sample_ids"] for r in process_view["records"]] == [["B", "C"]]
    assert process_view["batches"][0]["complete"]
    assert [r["sample_ids"] for r in usage.read_usage(db, "run")["records"]][-1] == ["D"]


def test_out_of_order_and_sparse_gaps(db):
    for position in [6, 4, 1]:
        record(db, position, 0, ["A"])
    assert usage.missing(db, "run", 8) == [
        dict(process=0, start=2, end=3),
        dict(process=0, start=5, end=5),
        dict(process=0, start=7, end=8),
    ]
    for position in [5, 3, 2]:
        record(db, position, 0, ["A"])
    usage.commit(db, "run", "saved", 6)
    with pytest.raises(ExperimentError):
        usage.commit(db, "run", "saved", 5)
    assert usage.missing(db, "run", 6) == []


def test_search_uses_applied_per_sample_and_batch_conditions(db):
    record(db, 1, 0, ["A", "B"], details={"reproduction": [{"scale": 3}, {"scale": 12}]})
    record(db, 2, 0, ["A", "C"], details={"reproduction": {"scale": 4}})
    results = list(usage.search(db, "run", condition="scale", minimum=0, maximum=5))
    assert [(r["sample_id"], r["applied_value"], r["position"]) for r in results] == [
        ("A", 3, 1), ("A", 4, 2), ("C", 4, 2),
    ]
