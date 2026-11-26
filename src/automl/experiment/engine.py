"""Local process supervision. Every child must claim its committed launch before doing work."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import traceback

from .errors import ExperimentError


_ACTIVE = {"accepted", "running"}
_LOG_NAMES = ("stdout.log", "stderr.log")
_TAIL_BYTES = 16 * 1024

# The worker's immediate parent preserves its actual OS exit status even if the
# Python monitor disappears. All paths and arguments are passed as argv entries.
_RECORD_EXIT = """
receipt=$1
shift
"$@"
code=$?
printf '%s\\n' "$code" > "$receipt.tmp" && mv -f "$receipt.tmp" "$receipt"
exit "$code"
"""


def _process(pid):
    """Linux process start ticks distinguish a live child from a reused PID."""
    fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b") ", 1)[1].split()
    return int(fields[19]), fields[0] in (b"Z", b"X")


def _identity(pid):
    try:
        start, _ = _process(pid)
    except (OSError, IndexError, ValueError):
        start = None
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        namespace = os.readlink("/proc/self/ns/pid")
    except OSError:
        boot = namespace = None
    return {"pid": pid, "start": start, "boot": boot, "namespace": namespace}


def _alive(identity):
    """Return True/False for a verified process and None for an unknown identity."""
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 0:
        return None
    try:
        if identity.get("boot"):
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            if boot != identity["boot"]:
                return False
        if identity.get("namespace") and os.readlink("/proc/self/ns/pid") != identity["namespace"]:
            return None
    except OSError:
        return None
    try:
        start, zombie = _process(identity["pid"])
    except (FileNotFoundError, ProcessLookupError):
        try:
            os.kill(identity["pid"], 0)
        except ProcessLookupError:
            return False
        except OSError:
            pass
        return None
    except (OSError, ValueError, IndexError):
        return None
    if identity.get("start") is None:
        return None
    return not zombie and start == identity["start"]


def launch_owner():
    """Persist this in the same transaction that accepts a run."""
    return {"execution_host": socket.gethostname(), "launcher_process": _identity(os.getpid())}


def _run_dir(store, run):
    return store.root / "runs" / run["id"]


def _tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - _TAIL_BYTES))
            raw = stream.read(_TAIL_BYTES)
            # Replacement characters for binary or truncated UTF-8 can expand
            # the decoded output. Keep the diagnostic field byte-bounded too.
            text = raw.decode("utf-8", errors="replace")
            return text.encode()[-_TAIL_BYTES:].decode(errors="ignore")
    except FileNotFoundError:
        return ""


def _active_stage(stages):
    for state in ("failed", "running", "pending"):
        for name in ("train", "evaluate"):
            if stages.get(name) == state:
                return name
    return None


def _outcome(run, code=None, *, problem=None, cancelled=False):
    """Project the process result onto the last committed state of each stage."""
    stages = dict(run["stages"])
    reasons = dict(run.get("stage_reasons", {}))
    failure = run.get("worker_error") or {}
    affected = failure.get("stage") or _active_stage(stages)
    finished = {"succeeded", "skipped", "failed", "interrupted"}
    unfinished = any(value not in finished for value in stages.values())
    if cancelled:
        state, reason = "interrupted", "Stopped before execution"
    elif problem or code != 0 or unfinished or "failed" in stages.values():
        state = "failed"
        reason = failure.get("message") or reasons.get(affected) or problem
        if not reason:
            if code:
                reason = f"Worker exited with status {code}"
            elif "failed" in stages.values():
                reason = f"Stage {affected} failed"
            else:
                reason = "Worker exited without reporting every stage result"
    elif "interrupted" in stages.values():
        state, reason = "interrupted", "Stopped at a safe point"
    else:
        state, reason = "succeeded", None

    # Keep successful stages and their artifacts even after an abnormal exit.
    for name in ("train", "evaluate"):
        if stages.get(name) not in ("pending", "running"):
            continue
        if name == "evaluate" and stages[name] == "pending" and stages.get("train") in ("failed", "interrupted"):
            reason_train = "失敗した" if stages["train"] == "failed" else "中断された"
            stages[name] = "skipped"
            reasons[name] = f"前段の学習が{reason_train}ため"
        else:
            stages[name] = state
            reasons[name] = reason
    error = None
    if state == "failed":
        error = {
            "code": "launch_failed" if run["status"] == "accepted" else "execution_failed",
            "message": reason,
            "details": {"stage": affected, "exit_code": code},
        }
    return {**run, "status": state, "stages": stages, "stage_reasons": reasons,
            "state_confirmed": True, "error": error}


def _observed(store, run):
    if run["status"] not in _ACTIVE:
        return {**run, "state_confirmed": True}
    if run.get("execution_host") != socket.gethostname():
        return {**run, "state_confirmed": False}

    if run["status"] == "running":
        worker = _alive(run.get("worker_process"))
        if worker is True:
            return {**run, "state_confirmed": True}
        owner = run.get("recorder_process")
    else:
        worker = False
        owner = run.get("recorder_process") or run.get("monitor_process") or run.get("launcher_process")
    alive = _alive(owner)
    # Inspect the receipt AFTER checking its writer. If the writer just exited,
    # its atomic rename has finished. A receipt proves the worker actually exited.
    try:
        code = int((_run_dir(store, run) / "worker.exit").read_text())
    except FileNotFoundError:
        pass
    except OSError:
        return {**run, "state_confirmed": False}
    except ValueError:
        if alive is False and worker is False:
            return _outcome(run, problem="Malformed worker exit status")
        return {**run, "state_confirmed": False}
    else:
        return _outcome(run, code)

    if run["status"] == "running":
        if worker is None or alive is not False:
            return {**run, "state_confirmed": False}
        return _outcome(run, problem="Worker exited without a confirmed exit status")
    if alive is False:
        return _outcome(run, problem="Launch process ended before the worker claimed the run")
    return {**run, "state_confirmed": alive is True}


def _save_diagnostics(store, db, run, error, directory):
    run_id = run["id"]
    stdout, stderr = (_tail(directory / name) for name in _LOG_NAMES)
    diagnostics = {"error": error, "stdout_tail": stdout, "stderr_tail": stderr}
    failure = run.get("worker_error") or {}
    if failure.get("traceback"):
        diagnostics["traceback_tail"] = failure["traceback"]
    elif "Traceback (most recent call last):" in stderr:
        diagnostics["traceback_tail"] = stderr[stderr.rindex("Traceback (most recent call last):"):]
    data = json.dumps(diagnostics, ensure_ascii=False, indent=2).encode()
    artifact = store.artifacts.save(db, run_id, "diagnostics", {"diagnostics.json": data})
    store.event(db, run_id, "diagnostics_saved", {"artifact_id": artifact, "file": "diagnostics.json"})


def _persist(store, db, run, outcome):
    run_id = run["id"]
    for name, state in outcome["stages"].items():
        if state != run["stages"][name]:
            store.stage(db, run_id, name, state, reason=outcome["stage_reasons"].get(name))
    directory = _run_dir(store, run)
    if outcome["status"] == "failed":
        try:
            _save_diagnostics(store, db, run, outcome["error"], directory)
        except OSError as exc:
            store.event(db, run_id, "diagnostics_save_failed", {"reason": str(exc)})
    files = {name: directory / name for name in _LOG_NAMES if (directory / name).is_file()}
    if files:
        try:
            artifact = store.artifacts.save(db, run_id, "logs", files)
        except OSError as exc:
            paths = [str(path.relative_to(store.root)) for path in files.values()]
            store.event(db, run_id, "logs_save_failed", {"reason": str(exc), "paths": paths})
        else:
            store.event(db, run_id, "logs_saved", {"artifact_id": artifact})
    return store.finish(db, run_id, outcome["status"], error=outcome["error"])


def status(store, run_id, *, reconcile=False):
    with store.transaction() as db:
        run = store.run(db, run_id)
        observed = _observed(store, run)
    if not reconcile or store.read_only or observed["status"] == run["status"]:
        return observed
    # Re-read inside the write transaction. The monitor or another client may
    # already have committed the result and freed the branch.
    with store.transaction(write=True) as db:
        run = store.run(db, run_id)
        observed = _observed(store, run)
        if observed["status"] == run["status"]:
            return observed
        return _persist(store, db, run, observed)


def stop(store, run_id):
    with store.transaction(write=True) as db:
        run = store.run(db, run_id)
        if run["status"] not in _ACTIVE:
            raise ExperimentError("invalid_request", "The run has already finished",
                                  run_id=run_id, status=run["status"])
        if not run.get("stop_requested"):
            store.update_run(db, run_id, stop_requested=True)
            store.event(db, run_id, "stop_requested", {})
        if run["status"] == "accepted":
            # The worker must acquire this very transaction before it can claim
            # the run and invoke any user code.
            _persist(store, db, run, _outcome(run, cancelled=True))


def _command(role, store, run_id, source, tree):
    # -S prevents editable installs and .pth files from replacing the snapshot.
    # process_environment supplies the interpreter's dependency directories.
    return [sys.executable, "-S", "-u", "-c",
            "from automl.experiment.engine import _main; _main()", role,
            str(store.root.resolve()), run_id, str(source), str(tree)]


def fail_launch(store, db, run_id, reason):
    """Finalize preparation/launch failure before any worker has started."""
    run = store.run(db, run_id)
    if run["status"] == "accepted":
        _persist(store, db, run, _outcome(run, problem=reason))


def launch(store, run_id, *, source, tree):
    """Hand an accepted run to a monitor detached from the caller's session."""
    from .replay import process_environment

    source, tree = Path(source).resolve(), Path(tree).resolve()
    with store.transaction(write=True) as db:
        run = store.run(db, run_id)
        if run["status"] != "accepted" or run.get("monitor_process"):
            return
        directory = _run_dir(store, run)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "stdout.log").open("ab") as stdout, (directory / "stderr.log").open("ab") as stderr:
                monitor = subprocess.Popen(
                    _command("monitor", store, run_id, source, tree),
                    env=process_environment(tree, root=store.root), cwd=tree,
                    start_new_session=True, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                )
        except (OSError, ValueError) as exc:
            _persist(store, db, run, _outcome(run, problem=str(exc)))
            return
        store.update_run(db, run_id, execution_host=socket.gethostname(),
                         monitor_process=_identity(monitor.pid), monitor_pid=monitor.pid)
    # Long-lived API clients also reap their finished monitors.
    try:
        threading.Thread(target=monitor.wait, daemon=True).start()
    except RuntimeError:
        # The monitor already owns the run. Popen can reap it on a subsequent
        # launch if this client is shutting down or cannot start a thread.
        pass


def _monitor(store, run_id, source, tree):
    with store.transaction(write=True) as db:
        run = store.run(db, run_id)
        if run["status"] != "accepted" or run.get("monitor_pid") != os.getpid():
            return
        args = _command("worker", store, run_id, source, tree)
        receipt = _run_dir(store, run) / "worker.exit"
        try:
            recorder = subprocess.Popen(["/bin/sh", "-c", _RECORD_EXIT, "automl", str(receipt), *args])
        except OSError as exc:
            _persist(store, db, run, _outcome(run, problem=str(exc)))
            return
        store.update_run(db, run_id, recorder_process=_identity(recorder.pid))
    recorder.wait()
    status(store, run_id, reconcile=True)


def _worker(store, run_id, source, tree):
    with store.transaction(write=True) as db:
        run = store.run(db, run_id)
        if run["status"] != "accepted" or (run.get("recorder_process") or {}).get("pid") != os.getppid():
            return
        store.update_run(db, run_id, status="running", worker_process=_identity(os.getpid()), pid=os.getpid())
        store.event(db, run_id, "worker_started", {"pid": os.getpid()})
    try:
        from .worker import execute

        execute(store, run_id, source=source, tree=tree)
    except BaseException as exc:
        trace = traceback.format_exc()
        print(trace, file=sys.stderr, flush=True)
        with store.transaction(write=True) as db:
            run = store.run(db, run_id)
            failure = {"message": str(exc) or type(exc).__name__, "stage": _active_stage(run["stages"]),
                       "traceback": trace.encode(errors="replace")[-_TAIL_BYTES:].decode(errors="ignore")}
            store.update_run(db, run_id, worker_error=failure)
        sys.exit(1)


def _main():
    from .store import Store

    role, root_arg, run_arg, source_arg, tree_arg = sys.argv[1:]
    {"monitor": _monitor, "worker": _worker}[role](
        Store(Path(root_arg)), run_arg, Path(source_arg), Path(tree_arg)
    )
