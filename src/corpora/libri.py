"""LibriSpeech chapter transcripts and LibriTTS original/normalized text."""

from functools import lru_cache
from pathlib import Path

from .audio import HEADER, Parser, audio_paths, recordings


def pipe_table(path):
    with path.open(encoding="utf-8") as lines:
        records = (
            [field.strip() for field in line.split("|")]
            for line in lines
            if line.strip() and not line.lstrip().startswith(";")
        )
        return {fields[0]: fields for fields in records}


def libri_parser(tts: bool):
    extra = dict.fromkeys(
        (
            "official_subset",
            "speaker_id",
            "chapter_id",
            "utterance_id",
            "speaker_gender",
            "speaker_name",
            "chapter_title",
            "book_title",
            "transcript",
        ),
        "VARCHAR",
    )
    if tts:
        extra.update(
            dict.fromkeys(("normalized_text", "segment_start_id", "segment_end_id"), "VARCHAR")
        )

    def rows(root, readers):
        ext = "txt" if tts else "TXT"
        speakers = pipe_table(root / f"SPEAKERS.{ext}")
        chapters = pipe_table(root / f"CHAPTERS.{ext}")
        separator = "_" if tts else "-"

        @lru_cache(maxsize=128)
        def transcripts(directory):
            name = directory.parent.name + separator + directory.name
            path = directory / (name + (".trans.tsv" if tts else ".trans.txt"))
            result = {}
            if path.is_file():
                with path.open(encoding="utf-8") as lines:
                    for line in lines:
                        if tts:
                            key, original, normalized = line.rstrip("\r\n").split("\t", 2)
                            result[key] = (original, normalized)
                        else:
                            key, text = line.rstrip("\r\n").split(" ", 1)
                            result[key] = (text,)
            return result

        def attributes(path: Path):
            subset, speaker, chapter, _ = path.parts
            parts = path.stem.split(separator)
            if parts[:2] != [speaker, chapter]:
                raise ValueError(f"Unexpected utterance name: {path}")
            person = speakers.get(speaker)
            book = chapters.get(chapter)
            text = transcripts(root / path.parent).get(path.stem, (None, None) if tts else (None,))
            fields = (
                subset,
                speaker,
                chapter,
                path.stem,
                person[1] if person else None,
                person[4] if person else None,
                book[6] if book else None,
                book[7] if book else None,
                *text,
            )
            if tts:
                start, end = parts[2:]
                return (*fields, start, end)
            return fields

        subsets = sorted(
            p
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith(("dev-", "test-", "train-"))
        )
        paths = (audio for subset in subsets for audio in audio_paths(subset))
        return recordings(root, paths, readers, attributes)

    return Parser(HEADER | extra, rows)


librispeech = libri_parser(False)
libritts = libri_parser(True)
