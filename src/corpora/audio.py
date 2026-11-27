"""Bounded, parallel reads of WAV and FLAC metadata."""

import os
import struct
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import soundfile


@dataclass(frozen=True)
class Parser:
    columns: dict[str, str]
    rows: Callable[[Path, int], Iterable[tuple]]


HEADER = {
    "relative_path": "VARCHAR PRIMARY KEY",
    "sample_rate": "INTEGER",
    "frames": "BIGINT",
    "channels": "INTEGER",
}


def walk_error(error: OSError):
    raise error


def audio_paths(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for parent, dirs, files in os.walk(directory, onerror=walk_error):
        dirs[:] = sorted(
            d for d in dirs if not d.startswith(".") and d not in {"__MACOSX", "_archives"}
        )
        for name in sorted(files):
            if not name.startswith(".") and name.lower().endswith((".wav", ".flac")):
                yield Path(parent) / name


def stream_info(path: Path) -> tuple[int, int, int]:
    # Read only the header over the network filesystem. Less common WAV
    # chunk layouts are delegated to libsndfile below.
    with path.open("rb", buffering=0) as handle:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            except OSError:
                pass
        header = handle.read(44)
        if (
            len(header) == 44
            and header[:4] == b"RIFF"
            and header[8:16] == b"WAVEfmt "
            and header[36:40] == b"data"
        ):
            fmt_size, codec, channels, rate, _, align = struct.unpack_from("<IHHIIH", header, 16)
            data_size = int.from_bytes(header[40:44], "little")
            if fmt_size == 16 and codec in (1, 3) and channels and rate and align and data_size != 0xFFFFFFFF:
                if data_size > os.fstat(handle.fileno()).st_size - len(header):
                    raise ValueError(f"Truncated WAV data in {path}")
                return rate, data_size // align, channels
    if (
        header[:4] == b"fLaC"
        and len(header) >= 42
        and header[4] & 0x7F == 0
        and header[5:8] == b"\x00\x00\x22"
    ):
        packed = int.from_bytes(header[18:26], "big")
        rate, frames, channels = packed >> 44, packed & 0xFFFFFFFFF, ((packed >> 41) & 7) + 1
        if rate and frames:
            return rate, frames, channels
    info = soundfile.info(path)
    return info.samplerate, info.frames, info.channels


def reader_budget():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return min(cpus or 1, 64)


def parallel_rows(paths, read, readers):
    # Limit both queued work and results even for million-file corpora.
    paths = iter(paths)
    pending = deque()

    def read_batch(batch):
        return [read(path) for path in batch]

    with ThreadPoolExecutor(max_workers=readers) as pool:
        while True:
            while len(pending) < 2 * readers:
                batch = list(islice(paths, 32))
                if not batch:
                    break
                pending.append(pool.submit(read_batch, batch))
            if not pending:
                return
            yield from pending.popleft().result()


def recordings(root: Path, paths, readers: int, attributes=lambda path: ()):
    def read(path):
        rel = path.relative_to(root)
        return (rel.as_posix(), *stream_info(path), *attributes(rel))

    return parallel_rows(paths, read, readers)


def layout_parser(columns=None, attributes=lambda path: ()):
    def rows(root, readers):
        return recordings(root, audio_paths(root), readers, attributes)

    return Parser(HEADER | (columns or {}), rows)
