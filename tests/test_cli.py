"""Scaffold-level tests: prove the package imports and the CLI is wired up.

These exist so CI is green before any feature work begins, per the build
protocol. They are replaced by real coverage as components land.
"""

from typer.testing import CliRunner

from nanoscope import __version__
from nanoscope.cli import app

runner = CliRunner()


def test_version_command_reports_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_bare_invocation_shows_help_rather_than_erroring() -> None:
    result = runner.invoke(app, [])
    assert "nanoscope" in result.stdout
