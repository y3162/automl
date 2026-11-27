"""Build the MNIST corpus metadata index from the four uncompressed IDX files."""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import struct
import tempfile
from contextlib import closing
from pathlib import Path

from automl.paths import installed_corpus_link

CORPUS = Path("data/corpora/mnist")
INDEX = Path("data/work/mnist/metadata.sqlite3")
FILES = {
    "train": ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    "test": ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
}
HEIGHT = WIDTH = 28
IMAGE_BYTES = HEIGHT * WIDTH
IMAGE_HEADER = struct.Struct(">4I")
LABEL_HEADER = struct.Struct(">2I")


def _index_pair(db, corpus: Path, partition: str) -> int:
    images_name, labels_name = FILES[partition]
    with (corpus / images_name).open("rb") as images, (corpus / labels_name).open("rb") as labels:
        try:
            magic, count, height, width = IMAGE_HEADER.unpack(images.read(IMAGE_HEADER.size))
            label_magic, label_count = LABEL_HEADER.unpack(labels.read(LABEL_HEADER.size))
        except struct.error as exc:
            raise ValueError(f"Truncated MNIST {partition} IDX header") from exc
        if (magic, height, width) != (2051, HEIGHT, WIDTH) or label_magic != 2049:
            raise ValueError(f"Invalid MNIST {partition} IDX magic number or image dimensions")
        if count != label_count or not count:
            raise ValueError(f"Invalid MNIST {partition} image/label counts: {count} and {label_count}")
        if os.fstat(images.fileno()).st_size != IMAGE_HEADER.size + count * IMAGE_BYTES:
            raise ValueError(f"Wrong MNIST image file size in {images_name}")
        values = labels.read()
        if len(values) != count or max(values, default=0) > 9:
            raise ValueError(f"Wrong MNIST label count or digit in {labels_name}")
        for filename, stream in ((images_name, images), (labels_name, labels)):
            stream.seek(0)
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            db.execute("INSERT INTO files VALUES (?, ?, ?, ?)", (filename, partition, count, digest))
        db.executemany("INSERT INTO samples VALUES (?, ?, ?)",
                       ((partition, index, digit) for index, digit in enumerate(values)))
    return count


def setup_data(project: Path, corpus: Path) -> dict[str, int]:
    """Validate the IDX files before updating the local symlink and metadata."""
    corpus = corpus.resolve(strict=True)
    link, index = project / CORPUS, project / INDEX
    index.parent.mkdir(parents=True, exist_ok=True)
    with (index.parent / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(dir=index.parent) as work:
            staged_index = Path(work) / index.name
            with closing(sqlite3.connect(staged_index)) as db, db:
                db.execute("""CREATE TABLE files (
                    filename TEXT PRIMARY KEY, partition TEXT, sample_count INTEGER, sha256 TEXT
                ) WITHOUT ROWID""")
                db.execute("""CREATE TABLE samples (
                    partition TEXT, idx INTEGER, label INTEGER, PRIMARY KEY (partition, idx)
                ) WITHOUT ROWID""")
                counts = {partition: _index_pair(db, corpus, partition) for partition in FILES}
            with installed_corpus_link(link, corpus):
                os.replace(staged_index, index)
        return counts
