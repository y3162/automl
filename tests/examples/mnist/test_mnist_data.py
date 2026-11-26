import hashlib
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing

import numpy as np
import pytest

from examples.mnist.data import MnistDataset
from examples.mnist.prepare import CORPUS, FILES, INDEX, setup_data


def test_make_setup_data_relinks_from_dotenv_and_rebuilds_identical_metadata(mnist_project, tmp_path):
    repo = Path(__file__).resolve().parents[3]
    shutil.copy(repo / "Makefile", mnist_project)
    (mnist_project / "scripts").mkdir()
    shutil.copy(repo / "scripts/setup_data.py", mnist_project / "scripts")
    corpus = (mnist_project / CORPUS).resolve()
    relocated = shutil.copytree(corpus, tmp_path / "relocated corpus")
    (mnist_project / ".env").write_text(f"MNIST_ROOT={shlex.quote(str(relocated))}\n")
    index = mnist_project / INDEX
    original = index.read_bytes()

    subprocess.run(["make", "setup-data", f"VENV_PYTHON={sys.executable}"],
                   cwd=mnist_project, check=True,
                   env={**os.environ, "PYTHONPATH": str(repo / "src")})

    assert (mnist_project / CORPUS).is_symlink()
    assert (mnist_project / CORPUS).resolve() == relocated
    assert index.read_bytes() == original
    with closing(sqlite3.connect(index)) as db:
        assert dict(db.execute("SELECT partition, COUNT(*) FROM samples GROUP BY partition")) == {
            "train": 24, "test": 8,
        }
        digests = dict(db.execute("SELECT filename, sha256 FROM files"))
    assert digests == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in corpus.iterdir()}


@pytest.mark.parametrize(("filename", "offset", "value", "message"), [
    (FILES["train"][0], 0, b"\xff", "magic number"),
    (FILES["test"][1], 7, b"\x07", "image/label counts"),
    (FILES["test"][1], 8, b"\x0a", "digit"),
    (FILES["train"][0], 16, None, "image file size"),
])
def test_invalid_idx_preserves_previous_setup(mnist_project, tmp_path, filename, offset, value, message):
    link, index = mnist_project / CORPUS, mnist_project / INDEX
    initial = index.read_bytes()
    corpus = link.resolve()
    broken = shutil.copytree(corpus, tmp_path / "bad")
    with (broken / filename).open("r+b") as stream:
        stream.seek(offset)
        if value is None:
            stream.truncate()
        else:
            stream.write(value)
    with pytest.raises(ValueError, match=message):
        setup_data(mnist_project, broken)
    assert link.resolve() == corpus
    assert index.read_bytes() == initial


def test_failed_index_install_restores_original_link(mnist_project, tmp_path, monkeypatch):
    link, index = mnist_project / CORPUS, mnist_project / INDEX
    previous = link.readlink()
    initial = index.read_bytes()
    relocated = shutil.copytree(link, tmp_path / "new")
    replace = os.replace

    def fail_index_install(src, dst):
        if dst == index:
            raise OSError("index installation failed")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_index_install)
    with pytest.raises(OSError, match="index installation failed"):
        setup_data(mnist_project, relocated)
    assert link.readlink() == previous
    assert index.read_bytes() == initial


def test_dataset_split_membership_and_pixel_lookup_survive_repartition(mnist_project):
    initial = MnistDataset.build({"validation_size": 4}, root=mnist_project)
    first_id = initial.create_version("first")
    updated = MnistDataset.build({"validation_size": 8}, root=mnist_project)
    next_id = updated.create_version("second")
    assert first_id != next_id
    assert updated.create_version("first") == first_id
    (mnist_project / INDEX).unlink()

    version = updated.open_version(first_id)
    for split, partition, indices in (("train", "train", range(20)),
                                      ("validation", "train", range(20, 24)),
                                      ("test", "test", range(8))):
        assert [s.sample_id for s in version.iter_samples(split=split)] == [f"{partition}/{i}" for i in indices]
    sample = version.get_sample("train/23")
    assert sample.metadata["label"] == 3
    assert sample.data.shape == (1, 28, 28) and sample.data.dtype == np.float32
    np.testing.assert_allclose(sample.data, 75 / 255)
    for unknown_id in ("test/8", "train/01", "train/-1", "validation/0"):
        with pytest.raises(KeyError):
            version.get_sample(unknown_id)
    assert [s.sample_id for s in updated.open_version(next_id).iter_samples(split="validation")] == [
        f"train/{i}" for i in range(16, 24)
    ]
