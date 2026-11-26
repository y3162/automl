"""Prepare corpora configured in the project .env via make setup-data."""

import os
from pathlib import Path

from examples.mnist.prepare import CORPUS, INDEX, setup_data


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    directory = os.environ.get("MNIST_ROOT")
    if not directory:
        raise SystemExit("Set MNIST_ROOT in .env to the directory containing the four MNIST IDX files")
    try:
        counts = setup_data(project, project / Path(directory).expanduser())
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{CORPUS} -> {(project / CORPUS).resolve()}")
    print(f"{INDEX}: {counts['train']} training and {counts['test']} test images")
