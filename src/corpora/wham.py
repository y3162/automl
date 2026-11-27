import csv

from .audio import HEADER, Parser, audio_paths, recordings


def wham_rows(root, readers):
    directory = root / "high_res_wham"
    with (directory / "high_res_metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = {row["Filename"]: row for row in csv.DictReader(handle)}

    def attributes(path):
        row = metadata[path.name]
        return (
            row["File ID"],
            row["Location ID"],
            row["Location Day ID"],
            row["WHAM! Split"],
            row["Noise Band"],
            row["Reverberation Level"],
            row["L to R Width (cm)"],
        )

    return recordings(root, audio_paths(directory / "audio"), readers, attributes)


parser = Parser(
    HEADER
    | dict.fromkeys(
        (
            "file_id",
            "location_id",
            "location_day",
            "official_split",
            "noise_band",
            "reverberation_level",
            "microphone_spacing",
        ),
        "VARCHAR",
    ),
    wham_rows,
)
