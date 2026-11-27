import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from methods.asr_guided_se.v1.data import SNR_RECIPE, prepare_snr_manifest
from methods.asr_guided_se.v1.voicebank_recipe import source_path, source_roots


def test_local_recipe_reproduces_the_recorded_version_and_changes_only_the_shift(tmp_path):
    recipe = tmp_path / SNR_RECIPE
    recipe.parent.mkdir()
    recipe.symlink_to(Path(__file__).resolve().parents[2] / SNR_RECIPE)
    work = tmp_path / "data/mock/work"
    work.mkdir(parents=True)
    (tmp_path / "data/work").symlink_to(work, target_is_directory=True)

    minus15 = prepare_snr_manifest(tmp_path, -15.0)
    assert minus15.name == "voicebank_m15db.jsonl"
    # Original dataset version of the A0–A7 manifest, including sample order.
    expected_version = "b0d4e0bc29a3cbe3a951d07bb29a96c9b872a69ed9c647abf273d5c0c1b6f98c"
    assert hashlib.sha256(minus15.read_bytes()).hexdigest() == expected_version
    minus10 = prepare_snr_manifest(tmp_path, -10)
    assert minus10.name == "voicebank_m10db.jsonl"
    splits = Counter()
    with minus15.open() as a, minus10.open() as b:
        for original, shifted in zip(map(json.loads, a), map(json.loads, b), strict=True):
            assert original.pop("snr_shift_db") == -15
            assert shifted.pop("snr_shift_db") == -10
            assert original == shifted
            splits[original["split"]] += 1
    assert splits == {"train": 11572, "test": 824}


def test_source_paths_follow_corpus_links_and_replay_older_configurations(tmp_path):
    corpora = tmp_path / "data/corpora"
    corpora.mkdir(parents=True)
    legacy = corpora / "asr_guided_se_sources"
    for name, corpus_id in (("VCTK", "vctk_0_80"), ("DEMAND48", "demand_48khz"),
                            ("Babble", "babble"), ("SSN", "ssn")):
        original = tmp_path / "recordings" / name
        original.mkdir(parents=True)
        (corpora / corpus_id).symlink_to(original, target_is_directory=True)
        source = {"asset": f"corpora/{name}/clip.wav"}
        for root in (corpora, legacy, original.parent):
            assert source_path(source_roots(root), source).resolve() == original / "clip.wav"
