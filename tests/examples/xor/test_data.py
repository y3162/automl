import hashlib
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from examples.xor.components.catalog import json_bytes
from examples.xor.components.data import CsvDataset, InlineDataset, ScaleFeatures


FIXTURE = Path(__file__).resolve().parents[3] / "configs/examples/xor/samples.csv"


def test_fixed_xor_splits_are_balanced_and_disjoint(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_STORE", str(tmp_path / "data/mock/store"))
    shutil.copyfile(FIXTURE, tmp_path / "xor.csv")
    dataset = CsvDataset.build({"path": "xor.csv"}, root=tmp_path)
    version = dataset.open_version(dataset.create_version("first"))
    ids = set()
    for split, count in (("train", 320), ("validation", 96), ("test", 96)):
        rows = list(version.iter_samples(split=split))
        assert len(rows) == count
        assert sum(row.metadata["label"] for row in rows) == count // 2
        for row in rows:
            assert row.sample_id not in ids
            ids.add(row.sample_id)
            x, y = row.data
            assert row.metadata["label"] == ((x > 0) != (y > 0))
    assert next(version.iter_samples(split="test")).sample_id == "104"


def test_snapshot_reuse_backup_and_index_repair(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    csv = source / "xor.csv"
    header = "sample_id,x1,x2,label\n"
    csv.write_text(header + "0,1,1,0\n104,1,-1,1\n")
    store = tmp_path / "one/data/mock/store/test"
    monkeypatch.setenv("AUTOML_STORE", str(store))
    first = CsvDataset.build({"path": "xor.csv"}, root=source)
    a = first.create_version("stable")
    frozen = first.open_version(a)
    csv.write_text(header + "104,1,-1,0\n105,-1,1,1\n")
    second = CsvDataset.build({"path": "xor.csv", "sha256": None}, root=source)
    assert second.create_version("stable") == a
    b = second.create_version("new-selection")
    assert b != a
    assert frozen.get_sample("104").metadata["label"] == 1
    assert second.open_version(b).get_sample("104").metadata["label"] == 0
    with sqlite3.connect(frozen.catalog) as db:
        db.execute("UPDATE samples SET split='validation', offset=0 WHERE id='104' AND version=?", (a,))
    assert [s.sample_id for s in second.open_version(a).iter_samples(split="test")] == ["104"]

    restored = tmp_path / "two/data/mock/store/test"
    shutil.copytree(store, restored)
    monkeypatch.setenv("AUTOML_STORE", str(restored))
    csv.unlink()
    CsvDataset.verify_version({}, source, a)
    assert not (tmp_path / "two/data/mock/work").exists()
    replica = CsvDataset.build({"path": "xor.csv", "sha256": a}, root=source)
    assert replica.create_version("stable") == a
    assert list(replica.open_version(b).iter_samples(split="test"))[1].sample_id == "105"
    assert list(replica.open_version(a).iter_samples(split="test"))[0] == frozen.get_sample("104")


def test_csv_validation_before_version_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_STORE", str(tmp_path / "data/mock/store"))
    csv = tmp_path / "invalid.csv"
    csv.write_text("sample_id,x1,x2,label\n1,1,-1,1\n1,-1,1,1\n")
    with pytest.raises(ValueError, match="duplicate"):
        CsvDataset.resolve_settings({"path": "invalid.csv"}, root=tmp_path)
    csv.write_text("sample_id,x1,x2,label\n1,nan,-1,1\n")
    with pytest.raises(ValueError, match="finite"):
        CsvDataset.resolve_settings({"path": "invalid.csv"}, root=tmp_path)
    assert not (tmp_path / "data/mock/store").exists()


def test_generated_registration_retry_provenance_and_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOML_STORE", str(tmp_path / "data/mock/store"))
    dataset = InlineDataset.build({"splits": {"train": [[1, -1, 1]], "test": []}}, root=tmp_path)
    version = dataset.create_version("data")
    sample = dataset.open_version(version).get_sample("train/0")
    transform = ScaleFeatures(2.0)
    result = transform.apply(sample.data)
    content = result.data.tolist()
    recipe = {
        "input": [version, sample.sample_id],
        "transform": "examples.xor.components.data.ScaleFeatures",
        "transform_version": "1",
        "conditions": result.reproduction,
        "expected_sha256": hashlib.sha256(json_bytes(content)).hexdigest(),
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: dataset.register_generated("derived", recipe=recipe), range(4)))
    assert len(set(ids)) == 1
    sample_id = ids[0]
    assert dataset.registration_status("derived") == {"status": "succeeded", "sample_id": sample_id}
    saved = dataset.get_generated(sample_id)
    assert saved["content"] is None
    assert saved["recipe"] == recipe
    restored = transform.apply(sample.data, reproduction=saved["recipe"]["conditions"])
    assert dataset.verify_generated(sample_id, restored.data)
    assert not dataset.verify_generated(sample_id, [0, 0])
    with pytest.raises(ValueError, match="different"):
        dataset.register_generated("derived", content=[0, 0])
    with pytest.raises(ValueError, match="unavailable"):
        dataset.register_generated("missing", recipe={**recipe, "input": [version, "train/42"]})
    assert dataset.registration_status("missing")["status"] == "failed"

    bare_id = dataset.register_generated("bare", content=result.data)
    with pytest.raises(ValueError, match="recipe"):
        dataset.ensure_recipe(bare_id)
    dataset.ensure_recipe(bare_id, recipe=recipe)
    assert dataset.register_generated("bare", content=content) == bare_id
    assert dataset.get_generated(bare_id) == {"recipe": recipe, "content": content}
    InlineDataset.verify_version({}, tmp_path, version)
    with sqlite3.connect(dataset.catalog) as db:
        db.execute("UPDATE versions SET splits=? WHERE id=?", (b"{}", version))
    with pytest.raises(ValueError, match="checksum"):
        InlineDataset.verify_version({}, tmp_path, version)
