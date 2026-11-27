"""Locate the original 48 kHz recordings and render a VoiceBank SNR-shift pair."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample


CORPUS_IDS = {"VCTK": "vctk_0_80", "DEMAND48": "demand_48khz", "Babble": "babble", "SSN": "ssn"}


def source_roots(corpus):
    # Old runs used a single link to the parent of the original recordings.
    # A restored checkout may have only the individual links from make setup-data.
    if corpus.name == "asr_guided_se_sources" and not corpus.is_dir():
        corpus = corpus.parent
    roots = {}
    for name, corpus_id in CORPUS_IDS.items():
        registered = corpus / corpus_id
        roots[name] = registered if registered.is_dir() else corpus / name
    return roots


def source_path(roots, source):
    name, *parts = Path(source["asset"]).relative_to("corpora").parts
    return roots[name].joinpath(*parts)


def render_pair(roots, row, verify):
    def read(source):
        path = source_path(roots, source)
        verify(path, source["sha256"])
        values, rate = sf.read(path, start=source["start"], stop=source["end"], always_2d=True, dtype="float64")
        if rate != 48000 or len(values) != source["end"] - source["start"]:
            raise ValueError(f"Invalid 48 kHz source span in {path}")
        return values[:, source["channel"]]

    clean = read(row["sources"]["speech"]) * row["clean_scale"] * row["clean_gain"]
    noise = read(row["sources"]["noise"])
    noisy = clean + row["noise_gain"] * 10 ** (-row["snr_shift_db"] / 20) * noise

    def at_16k(audio):
        return resample(torch.from_numpy(audio.astype(np.float32)).view(1, -1), 48000, 16000).view(-1)

    return at_16k(clean), at_16k(noisy)
