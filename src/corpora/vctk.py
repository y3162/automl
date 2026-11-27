from functools import lru_cache

from .audio import HEADER, Parser, audio_paths, recordings


def vctk_parser(audio_dir):
    def rows(root, readers):
        speakers = {}
        with (root / "speaker-info.txt").open(encoding="utf-8") as lines:
            next(lines)
            for line in lines:
                fields = line.partition("(")[0].split(maxsplit=4)
                if not fields:
                    continue
                speaker, age, gender, accent, *region = fields
                speakers[speaker.removeprefix("p")] = (
                    int(age) if age.isdecimal() else None,
                    gender,
                    accent,
                    region[0].strip() if region else None,
                )

        @lru_cache(maxsize=256)
        def transcript(speaker, utterance):
            path = root / "txt" / speaker / (utterance + ".txt")
            return path.read_text(encoding="utf-8").strip() if path.is_file() else None

        def attributes(path):
            speaker, index, *microphone = path.stem.split("_")
            utterance = f"{speaker}_{index}"
            person = speakers.get(speaker.removeprefix("p"), (None,) * 4)
            return (
                speaker,
                utterance,
                microphone[0] if microphone else None,
                *person,
                transcript(speaker, utterance),
            )

        return recordings(root, audio_paths(root / audio_dir), readers, attributes)

    return Parser(
        HEADER
        | {
            "speaker_id": "VARCHAR",
            "utterance_id": "VARCHAR",
            "microphone": "VARCHAR",
            "speaker_age": "INTEGER",
            "speaker_gender": "VARCHAR",
            "speaker_accent": "VARCHAR",
            "speaker_region": "VARCHAR",
            "transcript": "VARCHAR",
        },
        rows,
    )
