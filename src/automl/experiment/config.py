"""Resolve experiment YAML and construct the components selected for an execution."""
from __future__ import annotations

import base64
import copy
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeVar

import yaml

from automl.contracts import Component
from automl.evaluator import Evaluator
from automl.trainer import Trainer

from .errors import ExperimentError

T = TypeVar("T", bound=Component)


class _ConfigLoader(yaml.SafeLoader):
    def compose_mapping_node(self, anchor):
        # Check local keys before PyYAML expands merge keys. Explicit overrides
        # of an anchor are valid, two keys written in one mapping are ambiguous.
        node = super().compose_mapping_node(anchor)
        seen = set()
        for key, _ in node.value:
            mark = key.start_mark
            if key.tag not in ("tag:yaml.org,2002:str", "tag:yaml.org,2002:merge"):
                raise ExperimentError(
                    "invalid_config", f"{mark.name}:{mark.line + 1}: configuration keys must be strings",
                    file=mark.name, line=mark.line + 1,
                )
            value = key.value
            if value in seen:
                raise ExperimentError(
                    "invalid_config", f"{mark.name}:{mark.line + 1}: duplicate key {value!r}",
                    file=mark.name, line=mark.line + 1, key=value,
                )
            seen.add(value)
        return node


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ExperimentError("invalid_config", f"{name} must be a mapping with string keys", field=name)
    return value


def _overlay(parent: dict, child: dict, path: tuple[str, ...] = ()) -> dict:
    if (len(path) == 2 and path[0] == "components" and "implementation" in child
            and child["implementation"] != parent.get("implementation")):
        return copy.deepcopy(child)
    result = parent.copy()
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(parent.get(key), dict):
            result[key] = _overlay(parent[key], value, (*path, key))
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None, source: str | Path, *, base: dict | None = None) -> dict:
    """Apply explicitly written overrides in order, starting at saved evaluation conditions."""
    source = Path(source).resolve()
    merged = copy.deepcopy(base) if base is not None else {}
    layers = []
    seen = set()
    current = source / path if path is not None else None
    while current is not None:
        try:
            current = current.resolve()
            if current in seen:
                raise ExperimentError("invalid_config", f"configuration inheritance cycle at {current}", file=str(current))
            seen.add(current)
            with current.open(encoding="utf-8") as stream:
                layer = _mapping(yaml.load(stream, Loader=_ConfigLoader), str(current))
            # Catch recursive anchors and unsupported scalar types before merging.
            json.dumps(layer, allow_nan=False)
            layers.append(layer)
            if "extends" not in layer:
                break
            parent = layer.pop("extends")
            if not isinstance(parent, str) or not parent.strip() or Path(parent).is_absolute():
                raise ExperimentError("invalid_config", f"{current}: extends requires a project-relative path",
                                      file=str(current), field="extends")
            current = source / parent
        except ExperimentError:
            raise
        except (OSError, RuntimeError, UnicodeError, ValueError, TypeError, yaml.YAMLError) as exc:
            raise ExperimentError("invalid_config", f"{current}: {exc}", file=str(current)) from exc
    for layer in reversed(layers):
        merged = _overlay(merged, layer)
    return merged


def load_implementation(reference: str) -> type[Component]:
    """Import a concrete Component subclass from its dotted class name."""
    if not isinstance(reference, str) or "." not in reference or not all(
        part.isidentifier() for part in reference.split(".")
    ):
        raise ExperimentError("invalid_config", f"expected dotted component class name, got {reference!r}")
    module_name, _, class_name = reference.rpartition(".")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except Exception as exc:
        raise ExperimentError("invalid_config", f"cannot load {reference}: {exc}", implementation=reference) from exc
    if not isinstance(cls, type) or not issubclass(cls, Component) or inspect.isabstract(cls):
        raise ExperimentError("invalid_config", f"{reference} must implement a concrete Component", implementation=reference)
    return cls


def resolve_config(raw: dict, source: str | Path) -> dict:
    """Apply defaults and ask the selected implementations to validate their contracts."""
    source = Path(source)
    if _mapping(raw, "configuration").keys() != {"components", "plan"}:
        raise ExperimentError("invalid_config", "configuration requires only plan and components")
    plan = raw["plan"]
    if plan not in (["train"], ["evaluate"], ["train", "evaluate"]):
        raise ExperimentError("invalid_config", "plan must be [train], [evaluate] or [train, evaluate]", field="plan")
    specs = _mapping(raw["components"], "components")
    implementations = {}
    resolved = {}
    for name, spec in specs.items():
        spec = _mapping(spec, f"components.{name}")
        if not name or "implementation" not in spec or spec.keys() - {"implementation", "settings"}:
            raise ExperimentError("invalid_config", f"invalid component specification for {name!r}", component=name)
        cls = load_implementation(spec["implementation"])
        settings = _mapping(spec.get("settings", {}), f"components.{name}.settings")
        try:
            settings = _mapping(cls.resolve_settings(copy.deepcopy(settings), source), f"{name}.settings")
            json.dumps(settings, allow_nan=False)
        except Exception as exc:
            raise ExperimentError("invalid_config", f"invalid settings for {name}: {exc}", component=name,
                                  implementation=spec["implementation"]) from exc
        implementations[name] = cls
        resolved[name] = {"implementation": spec["implementation"], "settings": settings}

    for stage, name, contract in (("train", "trainer", Trainer), ("evaluate", "evaluator", Evaluator)):
        if stage in plan and name not in implementations:
            raise ExperimentError("invalid_config", f"{stage} requires a {contract.__name__} named {name}", component=name)
        if name in implementations and not issubclass(implementations[name], contract):
            raise ExperimentError("invalid_config", f"{name} must implement {contract.__name__}", component=name)
    for name, cls in implementations.items():
        if name == "trainer" and "train" not in plan:
            continue
        try:
            cls.validate_components(implementations)
        except Exception as exc:
            raise ExperimentError("invalid_config", f"incompatible components for {name}: {exc}", component=name) from exc
    if "train" in plan and "validation" in implementations:
        metric = resolved["trainer"]["settings"].get("selection_metric")
        if metric is not None:
            validator = implementations["validation"]
            if not issubclass(validator, Evaluator):
                raise ExperimentError("invalid_config", "validation must implement Evaluator", component="validation")
            try:
                metrics = tuple(validator.selection_metrics(resolved["validation"]["settings"]))
            except Exception as exc:
                raise ExperimentError("invalid_config", f"cannot resolve selection metrics: {exc}", component="validation") from exc
            if metric not in metrics:
                raise ExperimentError("invalid_config", f"unknown selection_metric {metric!r}; available: {metrics}",
                                      component="trainer", field="selection_metric", available=list(metrics))
    return {"plan": list(plan), "components": resolved}


class Components:
    """A factory bound to a run's resolved settings and data path base."""

    def __init__(self, config: dict, root: str | Path, *, versions: dict | None = None):
        self.specs = config["components"]
        self.root = Path(root)
        self.versions = dict(versions or {})

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def create(self, name: str, expected: type[T], **arguments: Any) -> T:
        if name not in self.specs:
            raise ExperimentError("invalid_config", f"component {name!r} was not configured", component=name)
        spec = self.specs[name]
        cls = load_implementation(spec["implementation"])
        if not issubclass(cls, expected):
            raise ExperimentError("invalid_config", f"{name} must implement {expected.__name__}", component=name)
        return cls.build(copy.deepcopy(spec["settings"]), root=self.root, **arguments)


def validate_config(raw: dict, source: str | Path, *, code_source: str | Path | None = None,
                    models: dict | None = None, store_root: str | Path | None = None,
                    evaluation_input: str | None = None, resume_state: dict | None = None) -> tuple[dict, list]:
    """Validate inside the fixed code tree, without reusing imports from another run."""
    from .replay import process_environment

    source = Path(source).resolve()
    project = Path(code_source or source).resolve()
    env = process_environment(project, root=store_root)
    request = {"raw": raw, "source": str(source), "evaluation_input": evaluation_input}
    if models:
        from .state import encode_states

        request["states"] = base64.b64encode(encode_states(models, {}, {})).decode("ascii")
    if resume_state is not None:
        training = resume_state["training"]
        if isinstance(training, bytes):
            training = base64.b64encode(training).decode("ascii")
        request["resume"] = {"training": training, "progress": resume_state["progress"]}
    command = [sys.executable, "-P", "-S", "-m", "automl.experiment.validate"]
    try:
        result = subprocess.run(command, cwd=project, env=env, input=json.dumps(request, allow_nan=False),
                                capture_output=True, text=True, encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        raise ExperimentError("invalid_config", f"cannot start component validation: {exc}") from exc
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        reason = result.stderr.strip()[-4096:] or f"exit code {result.returncode}"
        raise ExperimentError("invalid_config", f"component validation failed: {reason}") from exc
    if result.returncode or "error" in response:
        error = response.get("error", {})
        raise ExperimentError(error.get("code", "invalid_config"), error.get("message", "component validation failed"),
                              **error.get("details", {}))
    return response["config"], response["inputs"]
