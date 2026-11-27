from .audio import Parser, audio_paths, parallel_rows, stream_info


def voicebank_rows(root, readers):
    def pairs():
        subsets = {
            p.name.partition("_")[2].removesuffix("_wav")
            for pattern in ("clean_*_wav", "noisy_*_wav")
            for p in root.glob(pattern)
        }
        for subset in sorted(subsets):
            clean = root / f"clean_{subset}_wav"
            noisy = root / f"noisy_{subset}_wav"
            clean_files = {p.relative_to(clean): p for p in audio_paths(clean)}
            noisy_files = {p.relative_to(noisy): p for p in audio_paths(noisy)}
            if clean_files.keys() != noisy_files.keys():
                difference = clean_files.keys() ^ noisy_files.keys()
                raise ValueError(f"Unpaired VoiceBank file in {subset}: {min(difference)}")
            conditions = {}
            with (root / f"log_{subset}.txt").open(encoding="utf-8") as lines:
                for line in lines:
                    key, noise, snr = line.split()
                    conditions[key] = (noise, float(snr))
            for name, path in clean_files.items():
                yield subset, path, noisy_files[name], conditions.get(path.stem, (None, None))

    def read(pair):
        subset, clean, noisy, (noise, snr) = pair
        text_path = root / f"{subset}_txt" / (clean.stem + ".txt")
        text = text_path.read_text(encoding="utf-8").strip() if text_path.is_file() else None
        return (
            clean.relative_to(root).as_posix(),
            noisy.relative_to(root).as_posix(),
            subset,
            clean.stem,
            clean.stem.partition("_")[0],
            noise,
            snr,
            text,
            *stream_info(clean),
            *stream_info(noisy),
        )

    return parallel_rows(pairs(), read, readers)


parser = Parser(
    {
        "clean_relative_path": "VARCHAR PRIMARY KEY",
        "noisy_relative_path": "VARCHAR",
        "official_subset": "VARCHAR",
        "utterance_id": "VARCHAR",
        "speaker_id": "VARCHAR",
        "noise_type": "VARCHAR",
        "snr_db": "DOUBLE",
        "transcript": "VARCHAR",
        "clean_sample_rate": "INTEGER",
        "clean_frames": "BIGINT",
        "clean_channels": "INTEGER",
        "noisy_sample_rate": "INTEGER",
        "noisy_frames": "BIGINT",
        "noisy_channels": "INTEGER",
    },
    voicebank_rows,
)
