"""Prepare corpora configured in the project .env via make setup-data."""

import os
import sqlite3
from pathlib import Path

from corpora.catalog import CATALOG
from corpora.index import setup_audio
from examples.mnist.prepare import CORPUS, INDEX, setup_data as setup_mnist


def configured_sources(project, env):
    variables = {"mnist": "MNIST_ROOT"}
    variables.update({name: variable for name, (variable, _) in CATALOG.items()})
    selected = env.get("CORPUS")
    if selected:
        if selected not in variables:
            raise ValueError(f"Unknown corpus ID {selected!r}. Available: {', '.join(variables)}")
        variable = variables[selected]
        if not env.get(variable):
            raise ValueError(f"Set {variable} in .env for {selected}")
        variables = {selected: variable}
    else:
        known = set(variables.values())
        unknown = sorted(key for key, value in env.items()
                         if key.startswith("CORPUS_") and value and key not in known)
        if unknown:
            raise ValueError(f"Unknown corpus variables in .env: {', '.join(unknown)}")

    sources = {
        name: project / Path(env[key]).expanduser()
        for name, key in variables.items() if env.get(key)
    }
    if not sources:
        raise ValueError("Set MNIST_ROOT or a CORPUS_* path in .env")
    return sources


def main():
    project = Path(__file__).resolve().parents[1]
    sources = configured_sources(project, os.environ)
    try:
        jobs = int(os.environ.get("JOBS", "4"))
    except ValueError:
        raise ValueError("JOBS must be a positive integer") from None
    if jobs < 1:
        raise ValueError("JOBS must be a positive integer")

    failures = []
    mnist = sources.pop("mnist", None)
    if mnist is not None:
        try:
            counts = setup_mnist(project, mnist)
        except (OSError, ValueError, sqlite3.Error) as exc:
            failures.append(f"mnist: {exc}")
        else:
            print(f"{CORPUS} -> {(project / CORPUS).resolve()}")
            print(f"{INDEX}: {counts['train']} training and {counts['test']} test images")
    try:
        setup_audio(project, sources, jobs)
    except (OSError, RuntimeError) as exc:
        failures.append(str(exc))
    if failures:
        raise RuntimeError("\n".join(failures))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
