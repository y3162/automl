"""Project work areas and links to external data."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def work_directory(store: Path, *, fallback: Path) -> Path:
    """Map store/<case> to work/<case>, keeping each caller's custom-root layout."""
    for parent in (store, *store.parents):
        if parent.name == "store":
            return parent.parent / "work" / store.relative_to(parent)
    return fallback


@contextmanager
def installed_corpus_link(link: Path, source: Path):
    """Point link at the verified source while its caller commits the index under lock."""
    previous = link.readlink() if link.is_symlink() else None
    if previous is None and link.exists():
        raise ValueError(f"Expected a symlink at {link}, found an existing local file or directory")
    if previous == source:
        yield
        return

    link.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=link.parent) as directory:
        staged = Path(directory) / "next"
        backup = Path(directory) / "previous"
        staged.symlink_to(source, target_is_directory=True)
        if previous is not None:
            backup.symlink_to(previous, target_is_directory=True)
        os.replace(staged, link)
        try:
            yield
        except BaseException:
            if previous is None:
                link.unlink()
            else:
                os.replace(backup, link)
            raise
