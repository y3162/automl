"""Small deterministic IDX corpus for both MNIST tasks."""

import struct

import pytest

from examples.mnist.prepare import FILES, setup_data


@pytest.fixture
def mnist_project(tmp_path, monkeypatch):
    source = tmp_path / "project"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for partition, count in (("train", 24), ("test", 8)):
        image_file, label_file = FILES[partition]
        labels = bytes(index % 10 for index in range(count))
        pixels = b"".join(bytes([digit * 25]) * (28 * 28) for digit in labels)
        (corpus / image_file).write_bytes(struct.pack(">4I", 2051, count, 28, 28) + pixels)
        (corpus / label_file).write_bytes(struct.pack(">2I", 2049, count) + labels)
    setup_data(source, corpus)
    monkeypatch.setenv("AUTOML_STORE", str(source / "data/mock/store"))
    return source
