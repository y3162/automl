import pytest

from corpora.catalog import CATALOG


def records(name, root):
    _, parser = CATALOG[name]
    return [dict(zip(parser.columns, row, strict=True)) for row in parser.rows(root, 2)]


@pytest.mark.parametrize(("name", "attributes"), [
    ("babble", {}),
    ("ssn", {}),
    ("demand", {"scene_code": "DKITCHEN", "scene_class": "domestic", "channel_label": "ch01"}),
    ("demand_48khz", {"scene_code": "DKITCHEN", "scene_class": "domestic", "channel_label": "ch01"}),
    ("dns_challenge_5", {"collection": "DKITCHEN"}),
    ("dns_challenge_2020", {"collection": "DKITCHEN"}),
    ("ms_snsd", {"collection": "speech"}),
    ("openslr_26", {"collection": "speech"}),
    ("openslr_28", {"collection": "DKITCHEN"}),
    ("musan", {"category": "speech", "source": "DKITCHEN"}),
])
def test_directory_metadata(name, attributes, tmp_path, write_audio):
    filename = "speech/DKITCHEN/ch01.wav"
    write_audio(tmp_path / filename, channels=2)
    write_audio(tmp_path / "_archives/unused.wav")
    assert records(name, tmp_path) == [{
        "relative_path": filename, "sample_rate": 16000, "frames": 16, "channels": 2, **attributes,
    }]


@pytest.mark.parametrize("name", ["librispeech", "libritts"])
def test_libri_chapter_and_transcript(name, tmp_path, write_audio):
    tts = name == "libritts"
    ext = "txt" if tts else "TXT"
    (tmp_path / f"SPEAKERS.{ext}").write_text("; header\n12 | F | test-clean | 1.0 | Reader\n")
    (tmp_path / f"CHAPTERS.{ext}").write_text("34 | 12 | 1.0 | test-clean | 9 | 8 | Chapter | Book\n")
    chapter = tmp_path / "test-clean/12/34"
    if tts:
        utterance = "12_34_000001_000002"
        path = write_audio(chapter / f"{utterance}.wav")
        (chapter / "12_34.trans.tsv").write_text(f"{utterance}\tAt 2 pm.\tat two pm\n")
    else:
        utterance = "12-34-0001"
        path = write_audio(chapter / f"{utterance}.flac")
        (chapter / "12-34.trans.txt").write_text(f"{utterance} AT TWO PM\n")
    row, = records(name, tmp_path)
    expected = {
        "relative_path": path.relative_to(tmp_path).as_posix(), "sample_rate": 16000, "frames": 16,
        "channels": 1, "official_subset": "test-clean", "speaker_id": "12", "chapter_id": "34",
        "utterance_id": utterance, "speaker_gender": "F", "speaker_name": "Reader",
        "chapter_title": "Chapter", "book_title": "Book", "transcript": "At 2 pm." if tts else "AT TWO PM",
    }
    if tts:
        expected.update(normalized_text="at two pm", segment_start_id="000001", segment_end_id="000002")
    assert row == expected


@pytest.mark.parametrize("name", ["vctk_0_80", "vctk_0_92", "vctk_rva"])
def test_vctk_recordings_share_utterance_text(name, tmp_path, write_audio):
    (tmp_path / "speaker-info.txt").write_text("ID AGE GENDER ACCENT REGION\n225 23 F English Surrey\n")
    text = tmp_path / "txt/p225/p225_001.txt"
    text.parent.mkdir(parents=True)
    text.write_text("A sentence.\n")
    if name == "vctk_0_80":
        write_audio(tmp_path / "wav48/p225/p225_001.wav", rate=48000)
        microphones = [None]
    else:
        microphones = ["mic1", "mic2"]
        for microphone in microphones:
            write_audio(tmp_path / f"wav48_silence_trimmed/p225/p225_001_{microphone}.flac", rate=48000)
    rows = records(name, tmp_path)
    assert [row["microphone"] for row in rows] == microphones
    for row in rows:
        assert (row["speaker_id"], row["speaker_age"], row["utterance_id"]) == ("p225", 23, "p225_001")
        assert (row["transcript"], row["sample_rate"], row["frames"]) == ("A sentence.", 48000, 16)


def test_voicebank_pairs_and_subset_annotations(tmp_path, write_audio):
    for subset in ("trainset_28spk", "testset"):
        write_audio(tmp_path / f"clean_{subset}_wav/p001_001.wav")
        write_audio(tmp_path / f"noisy_{subset}_wav/p001_001.wav", rate=48000, frames=48)
        (tmp_path / f"{subset}_txt").mkdir()
        (tmp_path / f"{subset}_txt/p001_001.txt").write_text(subset)
        (tmp_path / f"log_{subset}.txt").write_text("p001_001 bus 12.5\n")
    rows = records("voicebank_demand", tmp_path)
    assert [row["official_subset"] for row in rows] == ["testset", "trainset_28spk"]
    for row in rows:
        subset = row["official_subset"]
        assert row["clean_relative_path"] == f"clean_{subset}_wav/p001_001.wav"
        assert row["noisy_relative_path"] == f"noisy_{subset}_wav/p001_001.wav"
        assert (row["transcript"], row["noise_type"], row["snr_db"]) == (subset, "bus", 12.5)
        assert (row["clean_sample_rate"], row["clean_frames"]) == (16000, 16)
        assert (row["noisy_sample_rate"], row["noisy_frames"]) == (48000, 48)


def test_chime_track_and_microphone(tmp_path, write_audio):
    write_audio(tmp_path / "data/audio/16kHz/isolated/tr05_bus_simu/F01_001C0201_BUS.CH2.wav")
    row, = records("chime_3", tmp_path)
    assert (row["track"], row["official_split"], row["environment"], row["recording_type"]) == (
        "isolated", "tr05", "bus", "simu",
    )
    assert (row["speaker_label"], row["utterance_id"], row["mic_channel"]) == ("F01", "001C0201", "2")


def test_wham_audio_and_csv(tmp_path, write_audio):
    directory = tmp_path / "high_res_wham"
    write_audio(directory / "audio/one.wav", channels=2, rate=48000)
    (directory / "high_res_metadata.csv").write_text(
        "Filename,File ID,Location ID,Location Day ID,WHAM! Split,Noise Band,Reverberation Level,L to R Width (cm)\n"
        "one.wav,1,2,3,cv,medium,low,20\n"
    )
    row, = records("wham_48khz", tmp_path)
    assert (row["relative_path"], row["file_id"], row["official_split"], row["microphone_spacing"]) == (
        "high_res_wham/audio/one.wav", "1", "cv", "20",
    )
    assert (row["sample_rate"], row["channels"]) == (48000, 2)


def test_urgent_manifest_with_and_without_reference(tmp_path, write_audio):
    header = "id\tnoisy_path\tclean_path\ttranscript\tsnr_dB\n"
    for subset in ("test_blind", "validation"):
        write_audio(tmp_path / subset / "noisy.wav")
    blind = tmp_path / "test_blind/metadata.tsv"
    blind.write_text(header + "b\tnoisy.wav\t\t<not-available>\t\n")
    write_audio(tmp_path / "validation/clean.wav")
    (tmp_path / "validation/metadata.tsv").write_text(header + "v\tnoisy.wav\tclean.wav\tHello.\t10\n")
    blind_row, valid_row = records("urgent_2024", tmp_path)
    assert (blind_row["noisy_relative_path"], blind_row["clean_relative_path"], blind_row["transcript"]) == (
        "test_blind/noisy.wav", None, None,
    )
    assert (valid_row["clean_relative_path"], valid_row["transcript"], valid_row["snr_db"]) == (
        "validation/clean.wav", "Hello.", 10.0,
    )
    blind.write_text(header + "b\t../validation/noisy.wav\t\t\t\n")
    with pytest.raises(ValueError, match="Invalid URGENT corpus path"):
        records("urgent_2024", tmp_path)
