import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest

from corpora.catalog import CATALOG
from corpora.index import INDEX, setup_audio
from examples.mnist.prepare import FILES, INDEX as MNIST_INDEX


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def make_project(tmp_path):
    project = tmp_path / "user's project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "Makefile", project)
    shutil.copy(REPO / "scripts/setup_data.py", project / "scripts")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("CORPUS") and key not in {"MNIST_ROOT", "JOBS", "MAKEFLAGS"}}

    def make(*options):
        return subprocess.run(["make", "setup-data", f"VENV_PYTHON={sys.executable}", *options],
                              cwd=project, env=env, text=True, capture_output=True)

    return project, make


def test_config_template_matches_audio_catalog():
    template = (REPO / ".env.example").read_text()
    keys = {line.split("=")[0] for line in template.splitlines() if line.startswith("CORPUS_")}
    assert keys == {variable for variable, _ in CATALOG.values()}
    assert len(keys) == 19


def test_make_dispatches_and_rebuilds_only_selected_corpus(make_project, write_audio):
    project, make = make_project
    write_audio(project / "first corpus/before.wav")
    write_audio(project / "other/noise.wav")
    mnist = project / "digits"
    mnist.mkdir()
    for images, labels in FILES.values():
        (mnist / images).write_bytes(struct.pack(">4I", 2051, 1, 28, 28) + bytes(784))
        (mnist / labels).write_bytes(struct.pack(">2I", 2049, 1) + bytes([2]))
    env = project / ".env"
    env.write_text('CORPUS_BABBLE="first corpus"\nCORPUS_SSN=other\nCORPUS_MUSAN=\nMNIST_ROOT=digits\n')
    result = make("JOBS=2")
    assert result.returncode == 0, result.stderr
    assert (project / MNIST_INDEX).is_file()
    links = project / "data/corpora"
    assert (links / "mnist").resolve() == mnist
    assert (links / "babble").resolve() == project / "first corpus"
    assert not (links / "musan").exists()

    write_audio(project / "relocated/new.flac", frames=24)
    env.write_text("CORPUS_BABBLE=relocated\nCORPUS_SSN=missing\nMNIST_ROOT=missing\n")
    for _ in range(2):
        result = make("CORPUS=babble")
        assert result.returncode == 0, result.stderr
        assert (links / "babble").resolve() == project / "relocated"
        assert (links / "ssn").resolve() == project / "other"
        with duckdb.connect(str(project / INDEX), read_only=True) as db:
            assert db.execute("SELECT * FROM corpora.babble").fetchall() == [("new.flac", 16000, 24, 1)]
            assert db.execute("SELECT relative_path FROM corpora.ssn").fetchall() == [("noise.wav",)]


@pytest.mark.parametrize(("config", "options", "message"), [
    ("", [], "Set MNIST_ROOT or a CORPUS_*"),
    ("", ["CORPUS=librispeech"], "Set CORPUS_LIBRISPEECH"),
    ("", ["CORPUS=wrong"], "Unknown corpus ID"),
    ("CORPUS_WRONG=wrong\n", [], "CORPUS_WRONG"),
    ("CORPUS_BABBLE=missing\n", ["JOBS=0"], "JOBS must be a positive integer"),
])
def test_selection_errors(make_project, config, options, message):
    project, make = make_project
    (project / ".env").write_text(config)
    result = make(*options)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (project / INDEX).exists()


def test_failed_pair_keeps_old_link_and_table_and_publishes_other_corpus(tmp_path, write_audio):
    original = tmp_path / "original"
    for kind in ("clean", "noisy"):
        write_audio(original / f"{kind}_testset_wav/old.wav")
    (original / "log_testset.txt").write_text("old bus 10\n")
    setup_audio(tmp_path, {"voicebank_demand": original}, 1)

    moved = tmp_path / "relocated"
    write_audio(moved / "clean_testset_wav/new.wav")
    (moved / "noisy_testset_wav").mkdir()
    (moved / "log_testset.txt").write_text("new bus 10\n")
    write_audio(tmp_path / "noise/one.wav")
    with pytest.raises(RuntimeError, match="voicebank_demand: Unpaired"):
        setup_audio(tmp_path, {"voicebank_demand": moved, "ssn": tmp_path / "noise"}, 2)
    assert (tmp_path / "data/corpora/voicebank_demand").resolve() == original
    assert (tmp_path / "data/corpora/ssn").resolve() == tmp_path / "noise"
    with duckdb.connect(str(tmp_path / INDEX), read_only=True) as db:
        assert db.execute("SELECT utterance_id FROM corpora.voicebank_demand").fetchall() == [("old",)]
        assert db.execute("SELECT relative_path FROM corpora.ssn").fetchall() == [("one.wav",)]


def test_bad_header_empty_corpus_and_link_conflict_preserve_index(tmp_path, write_audio):
    old = write_audio(tmp_path / "old/one.wav").parent
    setup_audio(tmp_path, {"babble": old}, 1)
    new = tmp_path / "new"
    new.mkdir()
    link = tmp_path / "data/corpora/babble"
    with pytest.raises(RuntimeError, match="No corpus records"):
        setup_audio(tmp_path, {"babble": new}, 1)
    (new / "broken.wav").write_bytes(b"broken header")
    with pytest.raises(RuntimeError, match="broken.wav"):
        setup_audio(tmp_path, {"babble": new}, 1)
    with write_audio(new / "broken.wav").open("r+b") as stream:
        stream.truncate(44)
    with pytest.raises(RuntimeError, match="Truncated WAV data"):
        setup_audio(tmp_path, {"babble": new}, 1)
    assert link.readlink() == old

    (new / "broken.wav").unlink()
    write_audio(new / "two.wav")
    link.unlink()
    link.mkdir()
    with pytest.raises(RuntimeError, match="Expected a symlink"):
        setup_audio(tmp_path, {"babble": new}, 1)
    assert link.is_dir() and not link.is_symlink()
    with duckdb.connect(str(tmp_path / INDEX), read_only=True) as db:
        assert db.execute("SELECT relative_path FROM corpora.babble").fetchall() == [("one.wav",)]


def test_independent_make_processes_publish_both_tables(make_project, write_audio):
    project, make = make_project
    for name in ("babble", "ssn"):
        write_audio(project / name / f"{name}.wav")
    (project / ".env").write_text("CORPUS_BABBLE=babble\nCORPUS_SSN=ssn\n")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(make, ["CORPUS=babble", "CORPUS=ssn"]))
    for result in results:
        assert result.returncode == 0, result.stderr
    with duckdb.connect(str(project / INDEX), read_only=True) as db:
        for name in ("babble", "ssn"):
            assert db.execute(f"SELECT relative_path FROM corpora.{name}").fetchall() == [(f"{name}.wav",)]
