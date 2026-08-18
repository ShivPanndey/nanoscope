"""Command line entry point.

Subcommands are registered here as each component lands. Keeping this file
honest matters: a verb appears only once the thing behind it actually works.
"""

import hashlib
import os
from pathlib import Path
from typing import Annotated

import typer

from nanoscope import __version__
from nanoscope.data.prepare import (
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_SHARD_TOKENS,
    DEFAULT_VAL_FRACTION,
    prepare,
)
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


data_app = typer.Typer(
    help="Turn a corpus into memory-mapped token shards.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")


def _ensure_output_is_writable(output_dir: Path) -> None:
    """Raise before any work starts if `output_dir` cannot be written to.

    `prepare` creates `output_dir` (and any missing parents) itself via
    `mkdir(parents=True, exist_ok=True)`, so `output_dir` itself commonly
    does not exist yet -- that is the expected case, not an error. What has
    to be writable is its nearest existing ancestor, since that is what
    `mkdir(parents=True)` actually writes into; permission is inherited down
    from there.

    `typer.Option(..., writable=True)` cannot do this job: click's
    writability check only runs when the path already exists, which is
    false for the common case of a fresh output directory, so it silently
    never fires. Checked here instead, so an unwritable destination is a
    named `--output` error before `prepare` reads the whole corpus and
    tokenizer, not a bare `PermissionError` traceback after.

    Being writable is not sufficient on its own: `os.access` reports a
    regular file as writable, and `mkdir(parents=True)` under one dies with
    `NotADirectoryError`. The ancestor has to be a directory too. The walk
    stops at anything that exists *or is a dangling symlink*, since
    `Path.exists()` follows symlinks and would otherwise step straight past
    a broken one, whose `mkdir` fails with `FileExistsError`. Both were
    live: each produced an uncaught traceback and exit 1, after the whole
    corpus had been read and digested.
    """
    ancestor = output_dir
    while not (ancestor.exists() or ancestor.is_symlink()):
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise typer.BadParameter(f"{ancestor} is not a directory", param_hint="--output")
    if not os.access(ancestor, os.W_OK):
        raise typer.BadParameter(f"{ancestor} is not writable", param_hint="--output")


@data_app.command("prep")
def data_prep(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Corpus file, one document per line.",
        ),
    ],
    tokenizer_path: Annotated[
        Path,
        typer.Option(
            "--tokenizer",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Trained tokenizer file.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=False,
            help="Destination directory for shards and the manifest.",
        ),
    ],
    val_fraction: Annotated[
        float,
        typer.Option(
            "--val-fraction",
            min=0.0,
            max=1.0,
            help="Fraction of documents held out for validation.",
        ),
    ] = DEFAULT_VAL_FRACTION,
    seed: Annotated[
        int,
        typer.Option("--seed", min=0, help="Seed for the document-level train/val split."),
    ] = 0,
    shard_tokens: Annotated[
        int,
        typer.Option(
            "--shard-tokens", min=1, help="Token count per shard file before rolling over."
        ),
    ] = DEFAULT_SHARD_TOKENS,
    max_chunk_bytes: Annotated[
        int,
        typer.Option(
            "--max-chunk-bytes",
            min=1,
            help="Reject a document whose longest pre-token chunk exceeds this many bytes.",
        ),
    ] = DEFAULT_MAX_CHUNK_BYTES,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Cap the number of documents processed."),
    ] = None,
) -> None:
    """Tokenize `source` with `tokenizer_path`, split it by document into
    train and val, and write both splits' shards plus a manifest under
    `output_dir`.

    All pipeline logic lives in `nanoscope.data.prepare`; this verb only
    parses and validates arguments, calls it, and reports what it did.
    Every path is checked before any work starts, since a full run is
    expensive and a typo must not cost it: `--source` and `--tokenizer` must
    already exist and be files, `--output` must not already be an existing
    file, and `--output`'s nearest existing ancestor must be writable.
    """
    _ensure_output_is_writable(output_dir)
    try:
        manifest = prepare(
            source,
            tokenizer_path,
            output_dir,
            seed=seed,
            val_fraction=val_fraction,
            shard_tokens=shard_tokens,
            max_chunk_bytes=max_chunk_bytes,
            limit=limit,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    train_tokens = manifest.tokens_in("train")
    val_tokens = manifest.tokens_in("val")
    typer.echo(
        f"wrote {output_dir}: {len(manifest.shards)} shards, "
        f"{train_tokens} train tokens, {val_tokens} val tokens, "
        f"{train_tokens + val_tokens} total tokens, "
        f"seed {manifest.seed}, val_fraction {manifest.val_fraction}, "
        f"tokenizer sha256 {manifest.tokenizer_sha256}"
    )
