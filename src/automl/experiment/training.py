"""Run-scoped training, data use and checkpoint operations."""
from __future__ import annotations

import hashlib
import json
import math
from uuid import uuid4

from automl.dataset import Dataset, usage
from automl.model import Model

from .errors import ExperimentError
from .state import decode_states, encode_states, json_bytes


def _digest(state):
    if isinstance(state, bytes):
        return 'bytes:' + hashlib.sha256(state).hexdigest()
    return 'json:' + hashlib.sha256(json_bytes(state)).hexdigest()


def _training_order(progress, order):
    if order is not None:
        if type(order) is not int:
            raise ValueError('checkpoint order must be an integer')
        return order
    for key in ('step', 'update', 'rounds', 'round', 'epoch'):
        if type(progress.get(key)) is int:
            return progress[key]
    counters = [value for value in progress.values() if type(value) is int]
    if len(counters) == 1:
        return counters[0]
    raise ValueError('specify an integer order for checkpoints with task-specific progress')


class TrainingContext:
    """Training methods shared by run and evaluation contexts."""

    def open_dataset(self, components, name='dataset'):
        dataset = components.create(name, Dataset)
        version = components.versions.get(name)
        if version is None:
            owner = self.run_id if self.record['mode'] == 'evaluate' else self.record['branch']
            version = dataset.create_version(f'{owner}:{name}')
        opened = dataset.open_version(version)
        with self.store.transaction(write=True) as db:
            record = self.store.run(db, self.run_id)
            versions = dict(record.get('dataset_versions', {}))
            if name in versions and versions[name] != version:
                raise ExperimentError('invalid_request', 'Dataset version changed during a run', component=name)
            if name not in versions:
                versions[name] = version
                self.store.update_run(db, self.run_id, dataset_versions=versions)
        components.versions[name] = version
        self.log('dataset_opened', component=name, version=version)
        return version, opened

    def initial_states(self):
        if self.record.get('initial_checkpoint') is None:
            return None
        with self.store.transaction() as db:
            data = self.store.artifacts.read(db, self.record['input_checkpoint_artifact'], 'states')
        return decode_states(data)

    def prepare_model(self, components, name='model'):
        states = self.initial_states()
        if states is None:
            model = components.create(name, Model, mode='new')
        else:
            mode = 'initialize' if self.record['mode'] == 'train' else 'restore'
            models = states['models']
            if name in models:
                state = models[name]
            elif mode == 'initialize' and len(models) == 1:
                state = next(iter(models.values()))
            else:
                raise ValueError(f'Checkpoint contains no state for model {name!r}')
            retain = self.record.get('retain') if mode == 'initialize' else None
            model = components.create(name, Model, mode=mode, state=state, retain=retain)
            if model.initialization_report is not None:
                self.log('model_initialized', component=name, **model.initialization_report)
        self._model_name = name
        return model

    def _models(self, model_state, model_states, model_name):
        if model_state is not None and model_states is None:
            name = model_name if model_name is not None else getattr(self, '_model_name', 'model')
            return {name: model_state}
        if model_states is not None and model_state is None and model_name is None:
            return model_states
        raise ValueError('supply model_state (optionally model_name) or a named model_states mapping')

    def start_training(self, model_state=None, training_state=None, progress=None, *,
                       model_name=None, model_states=None):
        models = self._models(model_state, model_states, model_name)
        data = encode_states(models, training_state, progress)
        details = {'progress': progress, 'model_names': sorted(models)}
        with self.store.artifacts.stage(self.run_id, 'start_state', {'states': data}, details=details) as publish, \
                self.store.transaction(write=True) as db:
            record = self.store.run(db, self.run_id)
            if record['stages'].get('train') != 'pending':
                raise ExperimentError('invalid_request', 'Training is not awaiting its start state', run_id=self.run_id)
            artifact = publish(db)
            self.store.event(db, self.run_id, 'start_state_saved', {'artifact_id': artifact, **details})
            self.store.stage(db, self.run_id, 'train', 'running')
            self.store.artifacts.release(db, self.run_id)

    def add(self, **record):
        """Durably record one process's actual use of input data."""
        with self.store.transaction(write=True) as db:
            usage.add(db, self.run_id, **record)

    def verify_usage(self, usage_range):
        """Confirm all process positions required by the task are durably present."""
        if usage_range is None:
            raise ExperimentError('invalid_request', 'Training result is missing its actual usage range')
        with self.store.transaction() as db:
            missing = usage.missing(db, self.run_id, usage_range)
        if missing:
            raise ExperimentError('invalid_request', 'Incomplete training usage', missing=missing, usage_range=usage_range)

    def _candidate(self, request_id, args):
        if request_id is None:
            return 'C' + uuid4().hex
        # Reserve identity and content separately from publication so a failed
        # attempt can retry the same candidate but cannot rewrite its inputs.
        scope = 'checkpoint-proposal:' + self.run_id
        with self.store.transaction(write=True) as db:
            try:
                candidate = self.store.request(db, request_id, 'save', args, scope=scope)
            except ExperimentError as exc:
                original = exc.details.get('original_args')
                if original is not None:
                    exc.details['fields'] = sorted(key for key, value in args.items() if value != original.get(key))
                raise
            if candidate is None:
                candidate = 'C' + uuid4().hex
                self.store.receipt(db, request_id, 'save', args, candidate, scope=scope)
            return candidate

    def _previous_identical(self, db, data, details, order):
        previous = self.store.artifacts.last_checkpoint(db, self.run_id)
        if not previous or previous['checkpoint_state'] != 'retained' or previous['order'] != order:
            return None
        if previous['details'] != details or previous['files']['states']['sha256'] != hashlib.sha256(data).hexdigest():
            return None
        try:
            self.store.artifacts.path(db, previous['id'], 'states')
        except ExperimentError:
            return None
        return previous['id']

    def save_checkpoint(self, *, training_state, progress, usage_range, model_state=None, model_states=None,
                        model_name=None, order=None, request_id=None, score=None, metric=None, reuse_identical=False):
        """Publish the state and its exact usage boundary in one transaction."""
        if score is not None and (isinstance(score, bool) or not math.isfinite(score)):
            raise ValueError('checkpoint score must be a finite number')
        models = self._models(model_state, model_states, model_name)
        data = encode_states(models, training_state, progress)
        order = _training_order(progress, order)
        # Freeze metadata now. Later updates by a trainer must not enlarge the
        # usage range while this request waits for a concurrent writer.
        details = json.loads(json_bytes({'progress': progress, 'usage_range': usage_range, 'score': score,
                                         'metric': metric, 'model_names': sorted(models)}))
        args = None
        if request_id is not None:
            states = decode_states(data)
            args = {'model_state': {name: _digest(value) for name, value in states['models'].items()},
                    'training_state': _digest(states['training']), 'order': order, **details}
        candidate = self._candidate(request_id, args)
        scope = 'checkpoint-result:' + self.run_id

        def existing(db):
            published = self.store.request(db, request_id, 'save', args, scope=scope)
            if published is None and reuse_identical:
                published = self._previous_identical(db, data, details, order)
                if published is not None:
                    self.store.receipt(db, request_id, 'save', args, published, scope=scope)
            return published

        try:
            with self.store.transaction(write=True) as db:
                published = existing(db)
                if published is not None:
                    return published
                if not self.store.run(db, self.run_id)['training_started']:
                    raise ExperimentError('invalid_request', 'Save the training start state before checkpoints')
                missing = usage.missing(db, self.run_id, details['usage_range'])
                if missing:
                    raise ExperimentError('invalid_request', 'Checkpoint is missing training usage',
                                          missing=missing, usage_range=details['usage_range'])
            # Flush the payload without holding the database writer lock. Other
            # processes can report new usage or request a stop during this I/O.
            with self.store.artifacts.stage(self.run_id, 'checkpoint', {'states': data}, details=details,
                                            order=order, artifact_id=candidate) as publish, \
                    self.store.transaction(write=True) as db:
                published = existing(db)
                if published is not None:
                    return published
                checkpoint = publish(db)
                usage.commit(db, self.run_id, checkpoint, details['usage_range'])
                self.store.event(db, self.run_id, 'checkpoint', {'checkpoint_id': checkpoint, **details})
                self.store.receipt(db, request_id, 'save', args, checkpoint, scope=scope)
                return checkpoint
        except Exception as exc:
            self.log('checkpoint_failed', progress=details['progress'], checkpoint_id=candidate,
                     request_id=request_id, reason=str(exc))
            raise

    def best_checkpoint(self, *, direction='max', run=None):
        """Return the best usable scored checkpoint on the run's actual resume path."""
        if direction not in ('min', 'max'):
            raise ValueError('checkpoint score direction must be min or max')
        path = []
        ceiling = None
        with self.store.transaction() as db:
            owner = run or self.run_id
            while owner:
                record = self.store.run(db, owner)
                candidates = [item for item in self.store.artifacts.list(db, owner)
                              if item['kind'] == 'checkpoint' and item['checkpoint_state'] == 'retained'
                              and (ceiling is None or item['order'] <= ceiling) and item['details'].get('score') is not None]
                path.append(candidates)
                owner = record.get('resume_from')
                if owner:
                    point = record.get('resume_checkpoint') or record['initial_checkpoint']
                    ceiling = self.store.artifacts.get(db, point)['order']
            ranked = [item for segment in reversed(path) for item in segment]
            ranked.sort(key=lambda item: item['details']['score'], reverse=direction == 'max')
            for candidate in ranked:
                try:
                    self.store.artifacts.path(db, candidate['id'], 'states')
                except ExperimentError:
                    continue
                return candidate['id'], candidate['details']['score']
        return None
