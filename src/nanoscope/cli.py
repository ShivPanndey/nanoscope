"""Command line entry point.

Subcommands are registered here as each component lands. Keeping this file
honest matters: a verb appears only once the thing behind it actually works.
"""

import hashlib
from pathlib import Path
from typing import Annotated

import typer

from nanoscope import __version__
from nanoscope.tokenizer import Tokenizer, train
from nanoscope.tokenizer.vocab import DEFAULT_VOCAB_SIZE, FIRST_MERGE_ID

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


tokenizer_app = typer.Typer(
    help="Train the byte-level BPE tokenizer.",
    no_args_is_help=True,
)
app.add_typer(tokenizer_app, name="tokenizer")


@tokenizer_app.command("train")
def tokenizer_train(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input", exists=True, dir_okay=False, readable=True, help="Training corpus."
        ),
    ],
    output_path: Annotated[
        Path, typer.Option("--output", dir_okay=False, help="Destination JSON file.")
    ],
    vocab_size: Annotated[
        int,
        typer.Option("--vocab-size", min=FIRST_MERGE_ID + 1, help="Total vocabulary size."),
    ] = DEFAULT_VOCAB_SIZE,
) -> None:
    """Train a tokenizer on a corpus file and write it to disk.

    The corpus is read whole rather than streamed: chunking a stream at
    arbitrary boundaries splits pre-tokens across them and perturbs the pair
    counts. See section 10 of the tokenizer spec.
    """
    if not output_path.parent.is_dir():
        raise typer.BadParameter(
            f"directory {output_path.parent} does not exist", param_hint="--output"
        )
    corpus = input_path.read_bytes()
    digest = hashlib.sha256(corpus).hexdigest()
    merges = train(corpus, vocab_size)
    tokenizer = Tokenizer(merges, corpus_sha256=digest)
    tokenizer.save(output_path)
    typer.echo(
        f"wrote {output_path}: {len(merges)} merges, "
        f"vocab {tokenizer.vocab_size}, corpus sha256 {digest}"
    )
