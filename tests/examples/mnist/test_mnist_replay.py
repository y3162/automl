import shutil
from pathlib import Path
from zipfile import ZipFile

import yaml

from automl.experiment import Experiment
from examples.mnist.prepare import CORPUS, FILES, setup_data


def test_restored_run_requires_its_external_idx_version(mnist_project, tmp_path):
    configs = Path("configs/examples/mnist/classifier")
    repo = Path(__file__).resolve().parents[3]
    shutil.copytree(repo / configs, mnist_project / configs)
    (mnist_project / "small.yaml").write_text(yaml.safe_dump({
        "extends": (configs / "train.yaml").as_posix(),
        "components": {
            "dataset": {"settings": {"validation_size": 4}},
            "trainer": {"settings": {"epochs": 1, "limit": 8, "batch_size": 4}},
            "validation": {"settings": {"limit": 4}},
            "evaluator": {"settings": {"limit": 4}},
        },
    }))
    store = Path("data/mock/store")
    source = Experiment(mnist_project / store, source=mnist_project)
    run = source.start(source.branch(), "small.yaml")
    status = source.wait(run, timeout=60)
    assert status["status"] == "succeeded", status
    with ZipFile(source.root / "sources" / (status["source_id"] + ".zip")) as archive:
        assert not any(name.startswith("data/") for name in archive.namelist())

    restored = tmp_path / "restored"
    shutil.copytree(source.root, restored / store)
    relocated_corpus = tmp_path / "relocated-mnist"
    (mnist_project / CORPUS).resolve().rename(relocated_corpus)
    mnist_project.rename(tmp_path / "offline")
    restore = Experiment(restored / store, source=restored)

    missing = restore.verify(run)
    assert not missing["ready"]
    assert {issue["item"] for issue in missing["issues"]} == {"dataset_version"}
    setup_data(restored, relocated_corpus)
    assert restore.verify(run)["ready"]
    reevaluation = restore.start(restore.branch(), evaluate=run)
    result = restore.wait(reevaluation, timeout=60)
    assert result["status"] == "succeeded", result
    assert result["dataset_versions"] == status["dataset_versions"]

    # One changed pixel leaves valid IDX headers, sizes and the old work index.
    with (relocated_corpus / FILES["test"][0]).open("r+b") as stream:
        stream.seek(-1, 2)
        pixel = stream.read(1)[0]
        stream.seek(-1, 2)
        stream.write(bytes([pixel ^ 1]))
    damaged = restore.verify(reevaluation)
    assert not damaged["ready"]
    assert [(issue["item"], issue["recorded"]) for issue in damaged["issues"]] == [
        ("dataset_version", status["dataset_versions"]["dataset"]),
    ]
