"""Freeze the code, dependencies and input bytes needed to replay a run."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml

from .errors import ExperimentError


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256 identifier {value!r}")
    return value


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def manifest_digest(manifest: Mapping) -> str:
    """Fingerprint the full manifest for the accepted run's database record."""
    return hashlib.sha256(_json_bytes(manifest)).hexdigest()


def _source_entries(path: Path, name: Path, excluded, ancestors: frozenset[Path] = frozenset()):
    if (path.name.startswith(".") or path.name == "__pycache__"
            or path.name.endswith((".egg-info", ".pyc", ".pyo"))):
        return
    if path.is_dir():
        resolved = path.resolve()
        if resolved in excluded or (path / "pyvenv.cfg").exists():
            return
        if resolved in ancestors:
            raise ValueError(f"Loop in source tree at {name}")
        for child in sorted(path.iterdir()):
            yield from _source_entries(child, name / child.name, excluded, ancestors | {resolved})
    elif path.is_file():
        yield name, path
    else:
        raise ValueError(f"Unreadable or unsupported source entry at {name}")


def _namespace_package(path: Path, excluded) -> bool:
    """PEP 420 also permits importable root packages without __init__.py."""
    if (not path.name.isidentifier() or not path.is_dir()
            or path.name in {"data", "venvs", "node_modules", "build", "dist"}):
        return False
    for directory, children, filenames in os.walk(path):
        if Path(directory).resolve() in excluded or "pyvenv.cfg" in filenames:
            children.clear()
            continue
        if any(name.endswith(".py") and name[:-3].isidentifier() for name in filenames):
            return True
        children[:] = [name for name in children if name.isidentifier() and name != "__pycache__"]
    return False


def _project_files(project: Path, store: Path):
    """Include the working tree's active files, independent of Git availability."""
    excluded = {store, _work_directory(store)}
    for path in sorted(project.iterdir()):
        if (path.name == "src" or path.suffix == ".py" or (path / "__init__.py").is_file()
                or _namespace_package(path, excluded)):
            yield from _source_entries(path, Path(path.name), excluded)

    # A project may use the installed AutoML and XOR packages alongside its own
    # modules. Bundle those packages without adding the live checkout to sys.path.
    library = Path(__file__).resolve().parents[2]
    for package in ("automl", "examples"):
        bundled = library / package
        if (bundled.is_dir() and not (project / package).exists()
                and not (project / "src" / package).exists()):
            yield from _source_entries(bundled, Path("src") / package, excluded)


def capture_source(root: Path | str, source: Path | str) -> str:
    """Publish a deterministic archive and return its content address."""
    directory = Path(root) / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with zipfile.ZipFile(staging, "w") as bundle:
            for name, path in _project_files(Path(source).resolve(), Path(root).resolve()):
                member = zipfile.ZipInfo(name.as_posix())
                member.create_system = 3
                member.compress_type = zipfile.ZIP_DEFLATED
                executable = path.stat().st_mode & 0o111
                member.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
                with path.open("rb") as data, bundle.open(member, "w", force_zip64=True) as target:
                    shutil.copyfileobj(data, target)
        identity = _digest(staging)
        final = directory / (identity + ".zip")
        if not final.exists():
            os.replace(staging, final)
        elif _digest(final) != identity:
            raise ValueError(f"Stored source archive is damaged: {identity}")
        return identity
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ExperimentError("invalid_config", f"Could not freeze project source: {error}") from error
    finally:
        staging.unlink(missing_ok=True)


def _archive_path(root: Path, source_id: str) -> Path:
    path = root / "sources" / (_check_digest(source_id) + ".zip")
    if _digest(path) != source_id:
        raise ValueError(f"Source archive does not match its SHA-256: {source_id}")
    return path


def _archive_files(bundle: zipfile.ZipFile):
    names = set()
    for entry in bundle.infolist():
        path = PurePosixPath(entry.filename)
        if (entry.filename != path.as_posix() or path.is_absolute()
                or ".." in path.parts or not path.parts or entry.filename in names
                or not stat.S_ISREG(entry.external_attr >> 16)):
            raise ValueError(f"Invalid entry in project archive: {entry.filename!r}")
        names.add(entry.filename)
        yield entry


def _work_directory(root: Path) -> Path:
    # data/store and data/mock/store/<case> each have a sibling work area.
    if root.name == "store":
        return root.parent / "work" / "automl-source"
    for parent in root.parents:
        if parent.name == "store":
            return parent.parent / "work" / root.relative_to(parent) / "automl-source"
    return root.parent / "work" / root.name / "automl-source"


def _matches(directory: Path, bundle: zipfile.ZipFile, entries: list[zipfile.ZipInfo]) -> bool:
    if not directory.is_dir() or directory.is_symlink():
        return False
    files = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            return False
        if path.is_file():
            files.add(path.relative_to(directory).as_posix())
    if files != {entry.filename for entry in entries}:
        return False
    for entry in entries:
        path = directory / entry.filename
        if (path.stat().st_size != entry.file_size
                or bool(path.stat().st_mode & 0o111) != bool((entry.external_attr >> 16) & 0o111)):
            return False
        with bundle.open(entry) as original:
            if _digest(path) != hashlib.file_digest(original, "sha256").hexdigest():
                return False
    return True


def restore_source(root: Path | str, source_id: str) -> Path:
    """Return a verified import tree rebuilt solely from the transferable store."""
    root = Path(root).resolve()
    try:
        archive = _archive_path(root, source_id)
        work = _work_directory(root)
        work.mkdir(parents=True, exist_ok=True)
        target = work / source_id
        with (work / (source_id + ".lock")).open("a") as lock, zipfile.ZipFile(archive) as bundle:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = list(_archive_files(bundle))
            if _matches(target, bundle, entries):
                return target
            with tempfile.TemporaryDirectory(dir=work) as stage:
                tree = Path(stage) / "source"
                tree.mkdir()
                for entry in entries:
                    path = tree / entry.filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as inp, path.open("xb") as out:
                        shutil.copyfileobj(inp, out)
                    path.chmod((entry.external_attr >> 16) & 0o777)
                if target.exists() or target.is_symlink():
                    target.rename(Path(stage) / "previous")
                tree.rename(target)
        return target
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ExperimentError("invalid_config", f"Cannot restore project source: {error}",
                              source_id=source_id) from error


def process_environment(code_project: Path | str, *, root: Path | str | None = None) -> dict[str, str]:
    """Import only the frozen project and venv dependencies in a python -S child."""
    code = Path(code_project).resolve()
    dependencies = sysconfig.get_paths()
    paths = (str(code / "src"), str(code), dependencies["purelib"], dependencies["platlib"])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(dict.fromkeys(paths)),
               PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1")
    if root is not None:
        env["AUTOML_STORE"] = str(Path(root).resolve())
    return env


def _environment() -> dict[str, str | None]:
    versions = {"python": platform.python_version(), "implementation": platform.python_implementation()}
    for package in ("numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _retain_input(directory: Path, record: Mapping, project: Path) -> dict:
    fingerprint = _check_digest(record["sha256"])
    target = directory / fingerprint
    if target.exists():
        if _digest(target) != fingerprint:
            raise ValueError(f"Stored input does not match its SHA-256: {fingerprint}")
    else:
        origin = project / Path(record["path"])
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temporary:
            staging = Path(temporary.name)
        try:
            with origin.open("rb") as data, staging.open("wb") as copy:
                shutil.copyfileobj(data, copy)
            if _digest(staging) != fingerprint:
                raise ValueError(f"Dataset input changed during acceptance: {origin}")
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
    return {"component": record["component"], "sha256": fingerprint}


def build_manifest(source_id: str, config: Mapping, inputs: Sequence[Mapping] | None = None,
                   *, root: Path | str | None = None, source: Path | str | None = None) -> dict:
    """Pin the validated config, input blobs, dependency versions, and exact code."""
    retained = []
    try:
        if inputs:
            if root is None or source is None:
                raise ValueError("A store and project are required to retain dataset input files")
            directory = Path(root) / "inputs"
            directory.mkdir(parents=True, exist_ok=True)
            retained = [_retain_input(directory, record, Path(source)) for record in inputs]
        return {
            "version": 1,
            "source_id": _check_digest(source_id),
            "config_sha256": hashlib.sha256(_json_bytes(config)).hexdigest(),
            "inputs": retained,
            "environment": _environment(),
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExperimentError("invalid_config", f"Cannot record replay conditions: {error}") from error


def _run_directory(root: Path | str, run_id: str) -> Path:
    if not isinstance(run_id, str) or run_id in ("", ".", "..") or Path(run_id).name != run_id:
        raise ExperimentError("invalid_request", "Invalid run identifier", run_id=run_id)
    return Path(root) / "runs" / run_id


def write_manifest(root: Path | str, run_id: str, manifest: Mapping) -> None:
    directory = _run_directory(root, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        path = Path(stream.name)
    try:
        path.write_bytes(_json_bytes(manifest) + b"\n")
        os.replace(path, directory / "replay.json")
    finally:
        path.unlink(missing_ok=True)


def read_manifest(root: Path | str, run_id: str) -> dict | None:
    path = _run_directory(root, run_id) / "replay.json"
    try:
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        return data
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise ExperimentError("invalid_request", f"Unreadable replay manifest: {error}",
                              run_id=run_id) from error


def _verify_versions(root, source_id, source, config, versions):
    """Execute the recorded Dataset's read-only hook with its frozen dependencies."""
    tree = restore_source(root, source_id)
    script = Path(__file__).with_name("verify_versions.py").resolve()
    request = {"source": str(Path(source).resolve()), "components": config["components"],
               "versions": versions}
    result = subprocess.run([sys.executable, "-S", "-P", str(script)], cwd=tree,
                            env=process_environment(tree, root=root), input=_json_bytes(request),
                            capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace")[-4096:] or "Version checker failed")
    return json.loads(result.stdout)


def verify_replay(root: Path | str, run_id: str, source: Path | str,
                  *, check_environment: bool = True, versions: Mapping | None = None,
                  expected_manifest: str | None = None) -> list[dict]:
    """Verify portable materials and optionally check the run's logical dataset versions."""
    root = Path(root)
    issues = []

    def report(item, reason, **details):
        issues.append({"run_id": run_id, "item": item, "reason": str(reason), **details})

    manifest_valid = True
    try:
        manifest = read_manifest(root, run_id)
        if manifest is None or manifest.get("version") != 1:
            raise ValueError("Replay manifest is missing or has an unsupported version")
        if expected_manifest is not None:
            actual_manifest = manifest_digest(manifest)
            if actual_manifest != expected_manifest:
                manifest_valid = False
                report("replay", "Replay manifest differs from the accepted run",
                       recorded=expected_manifest, current=actual_manifest)
    except (ValueError, ExperimentError) as error:
        report("replay", error)
        return issues

    source_valid = False
    try:
        archive = _archive_path(root, manifest["source_id"])
        with zipfile.ZipFile(archive) as bundle:
            list(_archive_files(bundle))
            if bad_entry := bundle.testzip():
                raise ValueError(f"Corrupted source entry {bad_entry}")
        source_valid = True
    except (TypeError, KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
        report("source_id", error)

    config_valid = False
    try:
        config = yaml.safe_load((_run_directory(root, run_id) / "config.yaml").read_bytes())
        expected = _check_digest(manifest["config_sha256"])
        actual = hashlib.sha256(_json_bytes(config)).hexdigest()
        if actual != expected:
            report("config_sha256", "Resolved config differs", recorded=expected, current=actual)
        else:
            config_valid = True
    except (TypeError, KeyError, ValueError, OSError, yaml.YAMLError) as error:
        report("config_sha256", error)

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        report("inputs", "Missing or malformed input list")
    else:
        for record in inputs:
            try:
                expected = _check_digest(record["sha256"])
                actual = _digest(root / "inputs" / expected)
                if actual != expected:
                    report("input", "Retained dataset bytes differ", recorded=expected, current=actual,
                           component=record.get("component"))
            except (TypeError, KeyError, ValueError, OSError) as error:
                report("input", error, input=record)

    if check_environment:
        saved = manifest.get("environment")
        if isinstance(saved, dict):
            for name, actual in _environment().items():
                if name not in saved or saved[name] != actual:
                    report(name, "Dependency environment differs", recorded=saved.get(name), current=actual)
        else:
            report("environment", "Missing dependency environment")
        if versions and source_valid and config_valid and manifest_valid:
            try:
                for issue in _verify_versions(root, manifest["source_id"], source, config, versions):
                    report("dataset_version", **issue)
            except (TypeError, KeyError, ValueError, OSError, ExperimentError) as error:
                report("dataset_version", error)
    return issues
