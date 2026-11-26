import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from automl.experiment.errors import ExperimentError
from automl.experiment.replay import (
    build_manifest,
    capture_source,
    manifest_digest,
    process_environment,
    read_manifest,
    restore_source,
    verify_replay,
    write_manifest,
)


@pytest.fixture
def workspace():
    mock = Path(__file__).resolve().parents[3] / "data" / "mock"
    (mock / "store").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=mock / "store") as name:
        stores = Path(name)
        work = mock / "work" / stores.name
        project = work / "project"
        for package in ("automl", "examples"):
            directory = project / "src" / package
            directory.mkdir(parents=True)
            (directory / "__init__.py").touch()
        try:
            yield stores, project
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_archived_sources_run_after_transfer_without_original_project(workspace, monkeypatch):
    stores, project = workspace
    (project / "helpers").mkdir()
    helper = project / "helpers" / "__init__.py"
    helper.write_text("def value(): return 42\n")
    namespace = project / "ns" / "nested"
    namespace.mkdir(parents=True)
    (namespace / "util.py").write_text("from helpers import value\n")
    (project / "runner.py").write_text("from ns.nested.util import value\n")
    first = capture_source(stores / "original", project)
    os.utime(helper)
    assert capture_source(stores / "original", project) == first
    helper.unlink()
    (project / "late.py").write_text("VALUE = 0\n")
    assert capture_source(stores / "original", project) != first

    restored_store = stores / "transfer"
    shutil.copytree(stores / "original", restored_store)
    shutil.rmtree(stores / "original")
    # Live source paths inherited from a calling shell must never fill holes
    # in a frozen snapshot (including Python's implicit cwd).
    monkeypatch.setenv("PYTHONPATH", str(project))
    tree = restore_source(restored_store, first)
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "from runner import value; from importlib.util import find_spec; "
         "assert find_spec('late') is None; print(value())"],
        cwd=project, env=process_environment(tree, root=restored_store),
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "42"
    (tree / "runner.py").write_text("damaged")
    (tree / "late.py").touch()
    assert restore_source(restored_store, first) == tree
    assert (tree / "runner.py").read_text() == "from ns.nested.util import value\n"
    assert not (tree / "late.py").exists()


def test_replay_pins_input_and_reports_all_integrity_and_environment_issues(workspace):
    stores, project = workspace
    run = "R1"
    config = {"plan": ["train"], "components": {}}
    source_id = capture_source(stores, project)
    csv = project / "sample.csv"
    csv.write_bytes(b"sample_id,x1,x2,label\n")
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    inputs = [{"component": "dataset", "path": csv, "sha256": digest}]
    manifest = build_manifest(source_id, config, inputs, root=stores, source=project)
    write_manifest(stores, run, manifest)
    run_files = stores / "runs" / run
    (run_files / "config.yaml").write_text(yaml.safe_dump(config))
    csv.unlink()
    assert read_manifest(stores, run) == manifest
    assert verify_replay(stores, run, project) == []
    accepted_digest = manifest_digest(manifest)
    write_manifest(stores, run, {**manifest, "inputs": []})
    altered = verify_replay(stores, run, project, expected_manifest=accepted_digest)
    assert [issue["item"] for issue in altered] == ["replay"]
    write_manifest(stores, run, manifest)
    assert len(list((stores / "inputs").iterdir())) == 1
    build_manifest(source_id, config, inputs, root=stores, source=project)
    (stores / "inputs" / digest).write_text("changed")
    (run_files / "config.yaml").write_text("plan: [evaluate]\n")
    manifest["environment"]["numpy"] = "different"
    (run_files / "replay.json").write_text(json.dumps(manifest))
    problems = {issue["item"]: issue for issue in verify_replay(stores, run, project)}
    assert set(problems) == {"input", "numpy", "config_sha256"}
    assert problems["numpy"]["recorded"] == "different"
    assert problems["numpy"]["current"]
    portable = verify_replay(stores, run, project, check_environment=False)
    assert {issue["item"] for issue in portable} == {"input", "config_sha256"}
    (stores / "sources" / (source_id + ".zip")).unlink()
    assert "source_id" in {issue["item"] for issue in verify_replay(stores, run, project)}


def test_first_capture_rejects_changed_input_without_publishing(workspace):
    stores, project = workspace
    csv = project / "sample.csv"
    csv.write_bytes(b"changed between validation and acceptance")
    stale = hashlib.sha256(b"previous").hexdigest()
    inputs = [{"component": "dataset", "path": csv, "sha256": stale}]
    with pytest.raises(ExperimentError, match="input changed"):
        build_manifest("1" * 64, {}, inputs, root=stores, source=project)
    assert not list((stores / "inputs").iterdir())


def test_logical_version_check_uses_dataset_from_original_source(workspace):
    store, project = workspace
    module = project / "reader.py"
    module.write_text(
        "class Reader:\n"
        "    @classmethod\n"
        "    def verify_version(cls, settings, root, version_id):\n"
        "        if version_id != settings['expected']:\n"
        "            raise ValueError('unknown fixed version')\n"
        "class BasicReader:\n"
        "    @classmethod\n"
        "    def verify_version(cls, settings, root, version_id):\n"
        "        raise NotImplementedError\n"
    )
    source_id = capture_source(store, project)
    module.write_text("raise RuntimeError('live code should never run')\n")
    config = {"components": {
        "dataset": {"implementation": "reader.Reader", "settings": {"expected": "fixed-id"}},
        "other": {"implementation": "reader.BasicReader", "settings": {}},
    }}
    manifest = build_manifest(source_id, config)
    write_manifest(store, "R1", manifest)
    (store / "runs" / "R1" / "config.yaml").write_text(yaml.safe_dump(config))
    assert verify_replay(store, "R1", project, versions={"dataset": "fixed-id", "other": "fixed"}) == []
    assert verify_replay(store, "R1", project, versions={"dataset": "lost-id"}) == [
        {"run_id": "R1", "item": "dataset_version", "reason": "unknown fixed version",
         "component": "dataset", "recorded": "lost-id"}
    ]
