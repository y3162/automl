import pytest

from automl.transform import Identity


def test_replay_preserves_payload_and_annotations():
    data = {"input": [1, 0], "class": 1}
    stage = Identity()
    first = stage.apply(data)
    assert first.data is data
    assert stage.apply(data, reproduction=first.reproduction).data is data
    with pytest.raises(ValueError):
        stage.apply(data, reproduction={"scale": 2})
