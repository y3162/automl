import csv
from pathlib import Path

from .audio import Parser, parallel_rows, stream_info


def urgent_rows(root, readers):
    def entries():
        for manifest in sorted(root.glob("*/metadata.tsv")):
            with manifest.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    yield manifest.parent.name, row

    def reference(subset, value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid URGENT corpus path: {value}")
        return Path(subset) / path

    def read(entry):
        subset, row = entry
        noisy = reference(subset, row["noisy_path"])
        clean = reference(subset, row["clean_path"]) if row.get("clean_path") else None
        if clean and not (root / clean).is_file():
            raise FileNotFoundError(root / clean)
        text = row.get("transcript")
        return (
            noisy.as_posix(),
            clean.as_posix() if clean else None,
            subset,
            row["id"],
            text if text != "<not-available>" else None,
            float(row["snr_dB"]) if row.get("snr_dB") else None,
            *stream_info(root / noisy),
        )

    return parallel_rows(entries(), read, readers)


parser = Parser(
    {
        "noisy_relative_path": "VARCHAR PRIMARY KEY",
        "clean_relative_path": "VARCHAR",
        "official_subset": "VARCHAR",
        "utterance_id": "VARCHAR",
        "transcript": "VARCHAR",
        "snr_db": "DOUBLE",
        "sample_rate": "INTEGER",
        "frames": "BIGINT",
        "channels": "INTEGER",
    },
    urgent_rows,
)
