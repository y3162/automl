from pathlib import Path

from .audio import HEADER, Parser, audio_paths, recordings


def chime_rows(root, readers):
    directory = Path("data/audio/16kHz")

    def attributes(path):
        track, *folders, _ = path.relative_to(directory).parts
        condition = folders[0] if folders else None
        split, scene, recording = (None, None, None)
        if condition:
            labels = condition.split("_")
            split = labels[0]
            if len(labels) == 3:
                scene, recording = labels[1:]
        stem, _, mic = path.stem.partition(".CH")
        tokens = stem.split("_")
        return (
            track,
            condition,
            split,
            scene,
            recording,
            tokens[0] if len(tokens) > 1 else None,
            ("_".join(tokens[1:-1]) if len(tokens) > 2 else tokens[-1]),
            mic or None,
        )

    return recordings(root, audio_paths(root / directory), readers, attributes)


parser = Parser(
    HEADER
    | dict.fromkeys(
        (
            "track",
            "condition",
            "official_split",
            "environment",
            "recording_type",
            "speaker_label",
            "utterance_id",
            "mic_channel",
        ),
        "VARCHAR",
    ),
    chime_rows,
)
