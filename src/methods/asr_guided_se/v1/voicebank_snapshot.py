"""Import audio-db VoiceBank SNR-shift recipes without depending on its experiment runtime."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample


def _strings(columns, name):
    data = columns[name + "_data"].tobytes()
    return [data[a:b].decode() for a, b in zip(columns[name + "_start"], columns[name + "_end"], strict=True)]


def snapshot_rows(snapshot, corpora):
    dataset = json.loads((snapshot / "dataset.json").read_bytes())
    if dataset["algorithm"] != "pregenerated.voicebank_snr_shift" or dataset["algorithm_version"] != 1:
        raise ValueError("Expected a VoiceBank SNR-shift v1 snapshot")
    bindings = {b["slot"]: b["case_set_id"] for b in dataset["bindings"]}
    with np.load(snapshot / "case_set_items.npz", allow_pickle=False) as items:
        order = list(zip(_strings(items, "case_set_id"), items["position"], _strings(items, "case_id"), strict=True))
    with (snapshot / "docs.jsonl").open() as file:
        docs = {r["case_id"]: r["doc_json"] for r in map(json.loads, file)}
    for split in ("train", "valid", "test"):
        selected = sorted((pos, uid) for group, pos, uid in order if group == bindings[split + ".cases"])
        for _, uid in selected:
            recipe = docs[uid]
            if recipe["algorithm"] != dataset["algorithm"] or recipe["sample_rate"] != 48000:
                raise ValueError(f"Unsupported recipe for {uid}")
            draws = recipe["draws"]
            text_path = corpora / "VCTK/txt" / draws["speaker_id"] / (draws["utterance_id"] + ".txt")
            yield dict(id=uid, split="test" if split == "test" else "train",
                       transcript=" ".join(text_path.read_text().split()) if text_path.is_file() else None,
                       sources=recipe["sources"], clean_scale=float(draws["clean_scale"]),
                       clean_gain=float(draws["clean_gain_g"]), noise_gain=float(draws["noise_gain_after_peak"]),
                       snr_shift_db=dataset["rule_json"]["snr_shift_db"])


def source_path(corpus, source):
    return corpus / Path(source["asset"]).relative_to("corpora")


def render_pair(corpus, row, verify):
    def read(source):
        path = source_path(corpus, source)
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
