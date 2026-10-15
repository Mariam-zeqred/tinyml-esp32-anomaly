"""PlatformIO custom test runner for test/minitest.h output.

Lines:  "  <test name>"            -> a new test case
        "FAIL file:line: ..."      -> the current case failed
        "N checks passed, M failed" -> end of run
"""

import click
from platformio.test.result import TestCase, TestStatus
from platformio.test.runners.base import TestRunnerBase


class CustomTestRunner(TestRunnerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = None
        self._failed = False
        self._message = ""

    def _flush(self):
        if self._name:
            case = TestCase(
                name=self._name,
                status=TestStatus.FAILED if self._failed else TestStatus.PASSED,
                message=self._message.strip() or None,
            )
            self.test_suite.add_case(case)
            if not self.options.verbose:
                click.echo(case.humanize())
        self._name, self._failed, self._message = None, False, ""

    def on_testing_line_output(self, line):
        if self.options.verbose:
            click.echo(line, nl=False)
        text = line.rstrip("\n")
        if text.startswith("  ") and not text.startswith("    "):
            self._flush()
            self._name = text.strip()
        elif text.startswith("FAIL ") or text.startswith("    "):
            self._failed = True
            self._message += text + "\n"
        elif "checks passed" in text:
            self._flush()
            self.test_suite.on_finish()
