"""Subprocess entry point for validating components from a specific code revision."""
import base64
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from automl.dataset import Dataset
from automl.model import Model
from automl.trainer import Trainer

from .config import Components, load_implementation, resolve_config
from .errors import ExperimentError


def _validate(request):
    source = Path(request["source"])
    resolved = resolve_config(request["raw"], source)
    components = Components(resolved, source)
    if request.get("evaluation_input") == "measurements":
        evaluator = load_implementation(components.specs["evaluator"]["implementation"])
        if not evaluator.uses_saved_measurements:
            raise ExperimentError("invalid_request", "evaluation requires a checkpoint or predictions",
                                  component="evaluator")
    if "resume" in request:
        trainer = components.create("trainer", Trainer)
        training = request["resume"]["training"]
        if isinstance(training, str):
            training = base64.b64decode(training, validate=True)
        try:
            trainer.validate_resume(training, request["resume"]["progress"])
        except Exception as exc:
            raise ExperimentError("invalid_request", f"cannot resume trainer: {exc}", component="trainer") from exc
    if "states" in request:
        from .state import decode_states

        states = decode_states(base64.b64decode(request["states"], validate=True))
        for name, state in states["models"].items():
            try:
                components.create(name, Model, mode="restore", state=state)
            except Exception as exc:
                raise ExperimentError("invalid_config", f"cannot restore model {name!r}: {exc}", component=name) from exc

    inputs = []
    for name, spec in components.specs.items():
        cls = load_implementation(spec["implementation"])
        if issubclass(cls, Dataset):
            try:
                for digest, path in cls.replay_files(spec["settings"], source).items():
                    inputs.append({"component": name, "sha256": digest, "path": str((source / path).resolve())})
            except Exception as exc:
                raise ExperimentError("invalid_config", f"cannot locate inputs for {name}: {exc}", component=name) from exc
    return {"config": resolved, "inputs": inputs}


def main():
    try:
        with redirect_stdout(sys.stderr):
            response = _validate(json.load(sys.stdin))
        print(json.dumps(response, allow_nan=False))
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, ExperimentError) else {"message": str(exc), "details": {}}
        print(json.dumps({"error": error}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
