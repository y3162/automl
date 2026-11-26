"""Portable checkpoint envelope for JSON and opaque binary component states."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from zipfile import BadZipFile, ZipFile, ZipInfo

from automl.contracts import State


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _validate(models, training, progress) -> None:
    if not isinstance(models, Mapping) or not models:
        raise ValueError("a checkpoint requires one or more named model states")
    if any(not isinstance(name, str) or not name for name in models):
        raise ValueError("model names must be nonempty strings")
    if any(not isinstance(state, (dict, bytes)) for state in models.values()):
        raise ValueError("model states must be dictionaries or bytes")
    if not isinstance(training, (dict, bytes)) or not isinstance(progress, dict):
        raise ValueError("a checkpoint requires a training state and a progress dictionary")


def encode_states(model_states: Mapping[str, State], training_state: State, progress: dict) -> bytes:
    """Freeze all state in one deterministic envelope, without deserializing binary state."""
    _validate(model_states, training_state, progress)
    header = {"models": dict(model_states), "training": training_state, "progress": progress}
    binary = {}
    for index, name in enumerate(sorted(model_states)):
        if isinstance(model_states[name], bytes):
            member = f"model-{index}.bin"
            binary[member] = model_states[name]
            header["models"][name] = member
    if isinstance(training_state, bytes):
        binary["training.bin"] = training_state
        header["training"] = "training.bin"

    encoded = json_bytes(header)
    if not binary:
        return encoded
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        # Fixed timestamps and entry order allow byte-for-byte comparison on retry.
        archive.writestr(ZipInfo("manifest.json"), encoded)
        for name, data in binary.items():
            archive.writestr(ZipInfo(name), data)
    return buffer.getvalue()


def decode_states(data: bytes) -> dict:
    try:
        if data.startswith(b"PK"):
            with ZipFile(io.BytesIO(data)) as archive:
                state = json.loads(archive.read("manifest.json"))
                for name, value in state["models"].items():
                    if isinstance(value, str):
                        state["models"][name] = archive.read(value)
                if isinstance(state["training"], str):
                    state["training"] = archive.read(state["training"])
        else:
            state = json.loads(data)
        _validate(state["models"], state["training"], state["progress"])
    except (BadZipFile, AttributeError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid checkpoint envelope: {exc}") from exc
    return state
