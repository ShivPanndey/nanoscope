"""Command line entry point.

Subcommands are registered here as each component lands. Keeping this file
honest matters: a verb appears only once the thing behind it actually works.
"""

import typer

from nanoscope import __version__

app = typer.Typer(
    name="nanoscope",
    help="A decoder-only transformer built from scratch.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Group callback.

    Without this, Typer collapses a single-command app into a bare command, so
    `nanoscope version` would parse "version" as a stray argument. Registering a
    callback keeps this a command group as further verbs are added.
    """


@app.command()
def version() -> None:
    """Print the installed nanoscope version."""
    typer.echo(__version__)
