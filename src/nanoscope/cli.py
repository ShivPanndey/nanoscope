"""Command line entry point.

Subcommands are registered here as each component lands. Keeping this file
honest matters: a verb appears only once the thing behind it actually works.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

import typer

from nanoscope import __version__
from nanoscope.data import prepare
from nanoscope.data.manifest import Manifest
from nanoscope.data.prepare import DEFAULT_MAX_CHUNK_BYTES
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


def _tokens_in_split(manifest: Manifest, split: Literal["train", "val"]) -> int:
    """Sum one split's token count from its shard entries.

    The manifest deliberately does not store per-split totals as a field of
    its own (see `nanoscope.data.manifest`'s docstring): a stored sum
    alongside the per-shard counts that produce it is redundant state that
    can drift out of sync. This is the one place that performs the sum, so
    nothing downstream re-derives it inline.
    """
    return sum(entry.tokens for entry in manifest.shards if entry.split == split)


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
    ] = 0.01,
    seed: Annotated[
        int,
        typer.Option("--seed", min=0, help="Seed for the document-level train/val split."),
    ] = 0,
    shard_tokens: Annotated[
        int,
        typer.Option(
            "--shard-tokens", min=1, help="Token count per shard file before rolling over."
        ),
    ] = 10_000_000,
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
    already exist and be files, and `--output` must not already be an
    existing file.
    """
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

    train_tokens = _tokens_in_split(manifest, "train")
    val_tokens = _tokens_in_split(manifest, "val")
    typer.echo(
        f"wrote {output_dir}: {len(manifest.shards)} shards, "
        f"{train_tokens} train tokens, {val_tokens} val tokens, "
        f"{train_tokens + val_tokens} total tokens, "
        f"seed {manifest.seed}, val_fraction {manifest.val_fraction}, "
        f"tokenizer sha256 {manifest.tokenizer_sha256}"
    )
