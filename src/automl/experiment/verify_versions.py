"""Isolated reader for logical dataset versions recorded by an experiment."""

import importlib
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def check_versions(request):
    source = Path(request["source"])
    issues = []
    for name, version in request["versions"].items():
        try:
            if not isinstance(version, str) or not version:
                raise ValueError("Missing dataset version identifier")
            spec = request["components"][name]
            module, _, attribute = spec["implementation"].rpartition(".")
            dataset = getattr(importlib.import_module(module), attribute)
            verify = getattr(dataset, "verify_version", None)
            if verify is not None:
                try:
                    verify(spec["settings"], source, version)
                except NotImplementedError:
                    # Some versions are fully described by the frozen project,
                    # settings and input blobs already checked by the caller.
                    pass
        except Exception as error:
            issues.append({"component": name, "recorded": version,
                           "reason": str(error) or type(error).__name__})
    return issues


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        issues = check_versions(json.load(sys.stdin))
    print(json.dumps(issues))
