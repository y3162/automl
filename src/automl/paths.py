"""Work areas paired with the project's store layout."""

from pathlib import Path


def work_directory(store: Path, *, fallback: Path) -> Path:
    """Map store/<case> to work/<case>, keeping each caller's custom-root layout."""
    for parent in (store, *store.parents):
        if parent.name == "store":
            return parent.parent / "work" / store.relative_to(parent)
    return fallback
