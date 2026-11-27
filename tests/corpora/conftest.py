import pytest
import soundfile


@pytest.fixture
def write_audio():
    def write(path, *, rate=16000, frames=16, channels=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(path, [[0.0] * channels] * frames, rate, subtype="PCM_16")
        return path

    return write
