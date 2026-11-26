import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from automl import cli


def invoke(*args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = cli.main(args)
    return result, stdout.getvalue(), stderr.getvalue()


class CliTests(unittest.TestCase):
    def test_parser_error_is_json_without_opening_store(self):
        with patch.object(cli, "Experiment") as factory:
            for args in [
                ["start", "xor.yaml"],
                ["read-results", "R", "E", "--limit", "3", "--after", "M", "--cursor", "abc"],
                ["wait", "R", "--timeout", "nan"],
            ]:
                with self.subTest(args=args):
                    code, stdout, stderr = invoke(*args)
                    self.assertEqual((code, stdout), (1, ""))
                    error = json.loads(stderr)
                    self.assertEqual(error["code"], "invalid_request")
                    self.assertTrue(error["message"])
                    self.assertIsInstance(error["details"], dict)
            factory.assert_not_called()

    def test_status_and_wait_report_run_outcome_on_stdout(self):
        with patch.object(cli, "Experiment") as factory:
            experiment = factory.return_value
            for command, state, expected in [
                ("status", "failed", 0),
                ("wait", "failed", 1),
                ("wait", "running", 2),
                ("wait", "interrupted", 0),
                ("wait", "succeeded", 0),
            ]:
                with self.subTest(command=command, state=state):
                    record = {"id": "R", "status": state}
                    getattr(experiment, command).return_value = record
                    args = ["--root", "scratch", command, "R"]
                    if command == "wait":
                        args.extend(["--timeout", "0"])
                    code, stdout, stderr = invoke(*args)
                    self.assertEqual((code, json.loads(stdout), stderr), (expected, record, ""))
                    factory.assert_called_with("scratch", read_only=False)
                    if command == "wait":
                        experiment.wait.assert_called_with(run="R", timeout=0)
                    else:
                        experiment.status.assert_called_with("R", reconcile=True)

    def test_read_aggregate_inputs_passes_page_options(self):
        with patch.object(cli, "Experiment") as factory:
            method = factory.return_value.read_aggregate_inputs
            method.return_value = {"inputs": [], "next_cursor": None}
            code, stdout, stderr = invoke("read-aggregate-inputs", "R", "E", "A", "--limit", "5",
                                         "--cursor", "page-token", "--root", "scratch")
            self.assertEqual((code, json.loads(stdout), stderr), (0, method.return_value, ""))
            factory.assert_called_once_with("scratch", read_only=True)
            method.assert_called_once_with(run_id="R", evaluation_id="E", aggregate_id="A",
                                           cursor="page-token", limit=5)

    def test_verify_exit_code_distinguishes_issues_and_pending_deletions(self):
        with patch.object(cli, "Experiment") as factory:
            for issues, expected in [([], 0), ([{"artifact_id": "A", "reason": "missing"}], 1)]:
                with self.subTest(issues=issues):
                    report = {"issues": issues, "pending_deletions": ["C"]}
                    factory.return_value.verify.return_value = report
                    code, stdout, stderr = invoke("verify")
                    self.assertEqual((code, json.loads(stdout), stderr), (expected, report, ""))
