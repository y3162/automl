"""Build each audio corpus in a private DuckDB file before publishing its table and link."""

import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from automl.paths import installed_corpus_link
from .audio import reader_budget
from .catalog import CATALOG

INDEX = Path("data/work/corpus_index.duckdb")


def _create_table(db, name):
    parser = CATALOG[name][1]
    columns = ", ".join(f'"{key}" {spec}' for key, spec in parser.columns.items())
    table = f'corpora."{name}"'
    db.execute("CREATE SCHEMA IF NOT EXISTS corpora")
    db.execute(f"CREATE OR REPLACE TABLE {table} ({columns})")
    return table


def _stage(path, name, source, readers):
    parser = CATALOG[name][1]
    arrays = ", ".join(f"unnest(?::{spec.split()[0]}[])" for spec in parser.columns.values())
    with duckdb.connect(str(path), config={"threads": 1}) as db:
        db.execute("BEGIN")
        table = _create_table(db, name)
        rows = iter(parser.rows(source, readers))
        count = 0
        while batch := list(islice(rows, 4096)):
            db.execute(f"INSERT INTO {table} SELECT {arrays}", list(zip(*batch, strict=True)))
            count += len(batch)
        if not count:
            raise ValueError(f"No corpus records found under {source}")
        db.commit()
    return count


def _publish(database, name, staged, link, source):
    # Independent make invocations can scan concurrently. Only the writer opens
    # the shared database, and holds the lock through commit and link rollback.
    with (database.parent / ".corpus-index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with duckdb.connect(str(database)) as db:
            # DuckDB 1.4 accepts only a string literal in ATTACH.
            filename = str(staged).replace("'", "''")
            db.execute(f"ATTACH '{filename}' AS staged (READ_ONLY)")
            db.execute("BEGIN")
            table = _create_table(db, name)
            db.execute(f"INSERT INTO {table} SELECT * FROM staged.{table}")
            with installed_corpus_link(link, source):
                db.commit()


def _setup_corpus(project, name, directory, readers):
    source = directory.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    database = project / INDEX
    link = project / "data/corpora" / name
    print(f"Indexing {name} from {source}", flush=True)
    with TemporaryDirectory(prefix="corpus-index-", dir=database.parent) as work:
        staged = Path(work) / "staged.duckdb"
        count = _stage(staged, name, source, readers)
        _publish(database, name, staged, link, source)
    return count


def setup_audio(project: Path, sources: dict[str, Path], jobs: int) -> None:
    """Index selected corpora, reporting each failure after the other jobs finish."""
    if jobs < 1:
        raise ValueError("JOBS must be a positive integer")
    if not sources:
        return
    (project / INDEX).parent.mkdir(parents=True, exist_ok=True)
    budget = reader_budget()
    workers = min(jobs, len(sources), budget)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {
            pool.submit(_setup_corpus, project, name, source, budget // workers): name
            for name, source in sources.items()
        }
        for task in as_completed(tasks):
            name = tasks[task]
            try:
                count = task.result()
            except Exception as exc:
                failures.append(f"{name}: {exc}")
            else:
                print(f"corpora.{name}: {count:,} rows", flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))
