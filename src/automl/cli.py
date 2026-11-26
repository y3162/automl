"""JSON command-line interface for local experiments."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections.abc import Sequence
from typing import Any

from automl.experiment import Experiment, ExperimentError


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ExperimentError("invalid_request", message)


def _timeout(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("timeout must be a finite, nonnegative number of seconds")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="automl", description=__doc__)
    parser.add_argument("--root", default="data/store", help="experiment data directory (default: data/store)")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, description: str, *, read_only: bool = False) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=description, description=description)
        # Accept the shared option on either side of the subcommand.
        sub.add_argument("--root", default=argparse.SUPPRESS, help="experiment data directory")
        sub.set_defaults(read_only=read_only)
        return sub

    branch = command("branch", "Create a branch.")
    branch.add_argument("--from-run", dest="origin_run", metavar="RUN")
    branch.add_argument("--checkpoint", metavar="REF")
    branch.add_argument("--request-id")

    start = command("start", "Start training or evaluation.")
    start.add_argument("config", nargs="?", metavar="CONFIG")
    destination = start.add_mutually_exclusive_group(required=True)
    destination.add_argument("--branch")
    destination.add_argument("--fork", metavar="REF", help="create a branch initialized from the checkpoint")
    inputs = start.add_mutually_exclusive_group()
    inputs.add_argument("--initialize", metavar="REF")
    inputs.add_argument("--evaluate", metavar="REF")
    inputs.add_argument("--predictions", metavar="PATH")
    inputs.add_argument("--prediction-artifact", metavar="ID")
    start.add_argument("--retain", action="append", metavar="NAME", help="retain a named part of the model (repeatable)")
    start.add_argument("--current-source", action="store_true")
    start.add_argument("--request-id")

    resume = command("resume", "Resume training on a branch.")
    resume.add_argument("branch", metavar="BRANCH")
    resume.add_argument("--request-id")

    command("stop", "Request a safe stop.").add_argument("run", metavar="RUN")
    command("status", "Inspect the current run state.").add_argument("run", metavar="RUN")
    wait = command("wait", "Wait for a run to finish.")
    wait.add_argument("run", metavar="RUN")
    wait.add_argument("--timeout", metavar="SECONDS", type=_timeout, help="maximum wait in seconds")

    command("delete-checkpoint", "Delete a checkpoint.").add_argument("checkpoint_id", metavar="CHECKPOINT_ID")

    decision = command("record-decision", "Record a decision based on an evaluation.")
    decision.add_argument("run", metavar="RUN")
    decision.add_argument("evaluation", metavar="EVALUATION")
    decision.add_argument("decision", metavar="DECISION")
    decision.add_argument("--aggregate-id")
    decision.add_argument("--followup-run")
    decision.add_argument("--request-id")

    history = command("history", "Read branch, run, event and artifact records.", read_only=True)
    history.add_argument("--run")
    history.add_argument("--summary", action="store_true")
    command("verify", "Verify saved data and reproducibility.", read_only=True).add_argument("--run")

    for name, description, identifiers in (
        ("list-evaluations", "List the evaluations of a run.", ("run_id",)),
        ("get-evaluation", "Read one evaluation.", ("run_id", "evaluation_id")),
        ("list-derived-evaluations", "List direct continuations, corrections and reaggregations.", ("run_id", "evaluation_id")),
        ("read-results", "Read a page of measurements.", ("run_id", "evaluation_id")),
        ("get-result", "Read one measurement.", ("run_id", "evaluation_id", "result_id")),
        ("list-aggregates", "List published aggregates.", ("run_id", "evaluation_id")),
        ("get-aggregate", "Read one aggregate.", ("run_id", "evaluation_id", "aggregate_id")),
        ("read-aggregate-inputs", "Read the direct inputs to an aggregate.", ("run_id", "evaluation_id", "aggregate_id")),
        ("list-uses", "List direct uses of a measurement or aggregate.", ("run_id", "evaluation_id")),
    ):
        query = command(name, description, read_only=True)
        for key in identifiers:
            query.add_argument(key, metavar=key.removesuffix("_id").upper())
        if name.startswith(("read-", "list-")):
            query.add_argument("--limit", required=True, type=int, metavar="N")
            position = query.add_mutually_exclusive_group()
            position.add_argument("--cursor")
            if name == "read-results":
                position.add_argument("--after", metavar="RESULT_ID")
        if name == "list-derived-evaluations":
            query.add_argument("--relation")
        if name == "list-uses":
            target = query.add_mutually_exclusive_group(required=True)
            target.add_argument("--result-id")
            target.add_argument("--aggregate-id")
    return parser


def _run(arguments: argparse.Namespace) -> tuple[Any, int]:
    options = vars(arguments).copy()
    command = options.pop("command")
    experiment = Experiment(options.pop("root"), read_only=options.pop("read_only"))
    if command == "wait":
        status = experiment.wait(**options)
        code = {"failed": 1, "accepted": 2, "running": 2}.get(status["status"], 0)
        return status, code
    if command == "status":
        return experiment.status(options["run"], reconcile=True), 0

    operation = getattr(experiment, command.replace("-", "_"))
    result = operation(**options)
    if command == "branch":
        return {"branch_id": result}, 0
    if command == "start":
        run_id = result["run_id"] if isinstance(result, dict) else result
        run = experiment.status(run_id)
        return {
            "run_id": run_id,
            "branch_id": run["branch"],
            "origin_run": run["origin_run"],
            "origin_checkpoint": run["origin_checkpoint"],
            "initial_checkpoint": run["initial_checkpoint"],
        }, 0
    if command == "resume":
        return {"run_id": result}, 0
    if command == "stop":
        return {"run_id": options["run"], "stop_requested": True}, 0
    if command == "record-decision":
        return {"decision_id": result}, 0
    if command == "verify":
        return result, int(bool(result["issues"]) or result.get("ready") is False)
    return result, 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result, exit_code = _run(_parser().parse_args(argv))
    except ExperimentError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error) as exc:
        error = ExperimentError("invalid_request", str(exc))
        print(json.dumps(error.as_dict(), ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
