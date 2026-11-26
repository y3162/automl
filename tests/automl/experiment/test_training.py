"""Checkpoint retries, data-use boundaries and binary state round trips."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import pytest

from automl.experiment import artifacts
from automl.experiment.errors import ExperimentError
from automl.experiment.runtime import RunContext
from automl.experiment.state import decode_states, encode_states
from automl.experiment.store import Store


@pytest.fixture
def training():
    mock_store = Path(__file__).resolve().parents[3] / 'data/mock/store'
    mock_store.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=mock_store, prefix='training-') as root:
        store = Store(root)
        with store.transaction(write=True) as db:
            run = store.create_run(db, store.create_branch(db), {'plan': ['train']}, mode='train')
        context = RunContext(store, run)
        context.start_training(model_states={'left': b'initial', 'right': {'w': 0}},
                               training_state=b'new', progress={'step': 0})
        yield context


def test_state_roundtrip_is_deterministic_and_preserves_opaque_states():
    left = {'a': {'b': 1}, 'right': b'\xff\x00'}
    right = dict(reversed(left.items()))
    first = encode_states(left, b'\xff', {'epoch': 7})
    assert first == encode_states(right, b'\xff', {'epoch': 7})
    assert decode_states(first) == {'models': left, 'training': b'\xff', 'progress': {'epoch': 7}}
    with pytest.raises(ValueError):
        decode_states(first[:100])


def test_missing_usage_and_concurrent_retries_publish_once(training):
    options = dict(model_states={'left': b'new', 'right': {'w': 2}}, training_state=b'shuffled',
                   progress={'step': 7}, usage_range={'0': 1, '1': 1}, request_id='save-7')
    training.add(process=1, position=1, usage_id='p1', sample_ids=['b'], dataset_version='v1')
    with pytest.raises(ExperimentError) as failure:
        training.save_checkpoint(**options)
    assert failure.value.details['missing'] == [{'process': 0, 'start': 1, 'end': 1}]
    assert all(item['kind'] != 'checkpoint' for item in training.store.history(training.run_id)['artifacts'])
    with pytest.raises(ExperimentError) as conflict:
        training.save_checkpoint(**{**options, 'usage_range': {'1': 1}})
    assert conflict.value.details['fields'] == ['usage_range']

    training.add(position=1, usage_id='p0', sample_ids=['a'], dataset_version='v1')
    training.add(position=2, usage_id='p0-next', sample_ids=['c'], dataset_version='v1')
    with ThreadPoolExecutor(max_workers=2) as executor:
        published = [executor.submit(training.save_checkpoint, **options) for _ in range(2)]
    ids = {future.result() for future in published}
    assert len(ids) == 1
    assert training.save_checkpoint(**options) in ids
    history = training.store.history(training.run_id)
    assert len(history['usage']['commits']) == 1
    assert len(history['usage']['records']) == 3
    checkpoint, = (item for item in history['artifacts'] if item['kind'] == 'checkpoint')
    assert checkpoint['id'] in ids
    assert checkpoint['details']['usage_range'] == {'0': 1, '1': 1}
    failed, = (event for event in history['events'] if event['kind'] == 'checkpoint_failed')
    assert failed['payload']['checkpoint_id'] == checkpoint['id']


def test_temporary_failure_preserves_request_and_reuse_checks_bytes(training, monkeypatch):
    options = dict(model_state=b'weights', training_state={'s': 1}, progress={'rounds': 0},
                   usage_range=0, score=0.5, reuse_identical=True)
    def no_space(*args, **kwargs):
        raise OSError('disk full')

    with monkeypatch.context() as patch:
        patch.setattr(artifacts, '_copy_content', no_space)
        with pytest.raises(OSError, match='disk full'):
            training.save_checkpoint(**options, request_id='save')
    checkpoint = training.save_checkpoint(**options, request_id='save')
    assert training.save_checkpoint(**options) == checkpoint
    with training.store.transaction() as db:
        path = training.store.artifacts.path(db, checkpoint, 'states')
    path.write_bytes(b'corrupt')
    replacement = training.save_checkpoint(**options)
    assert replacement != checkpoint
    assert training.best_checkpoint() == (replacement, 0.5)


def test_new_usage_can_arrive_during_checkpoint_write(training, monkeypatch):
    writing, release = Event(), Event()
    copy = artifacts._copy_content

    def slow_copy(content, destination):
        writing.set()
        assert release.wait(10)
        return copy(content, destination)

    monkeypatch.setattr(artifacts, '_copy_content', slow_copy)
    training.add(position=1, usage_id='first', sample_ids=['a'], dataset_version='v1')
    with ThreadPoolExecutor(max_workers=2) as executor:
        save = executor.submit(training.save_checkpoint, model_state=b'step1', training_state={},
                               progress={'step': 1}, usage_range=1)
        try:
            assert writing.wait(10)
            add = executor.submit(training.add, position=2, usage_id='next',
                                  sample_ids=['b'], dataset_version='v1')
            add.result(timeout=5)
            training.verify_usage(2)
        finally:
            release.set()
        checkpoint = save.result()
    with training.store.transaction() as db:
        assert training.store.artifacts.get(db, checkpoint)['details']['usage_range'] == 1
