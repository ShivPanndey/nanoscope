"""End-to-end test of the `data prep` verb."""

import re
from pathlib import Path

from typer.testing import CliRunner

from nanoscope.cli import app
from nanoscope.data.manifest import Manifest
from nanoscope.data.shards import ShardedTokens
from nanoscope.tokenizer import Tokenizer

runner = CliRunner()


_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _squash(text: str) -> str:
    """Collapse a Rich-rendered error panel into one lowercase alnum-only string.

    See `tests/test_tokenizer_cli.py`'s `_squash` for the full rationale: escape
    sequences must be stripped as whole sequences, before filtering on `isalnum`,
    or a colourised run leaves fragments like `36m` embedded in the middle of a
    word and the substring assertion silently passes for the wrong reason.
    """
    return "".join(ch for ch in _ANSI.sub("", text) if ch.isalnum()).lower()


def _write_tokenizer(path: Path) -> None:
    """A small, non-trivial hand-built tokenizer: one real merge (the byte
    pair for 'd' and 'o', both of which appear in the synthetic corpus
    below), not just the identity byte-level vocabulary."""
    Tokenizer(merges=[(ord("d"), ord("o"))]).save(path)


def _write_corpus(path: Path, doc_count: int) -> None:
    docs = [f"doc number {i} says hello".encode() for i in range(doc_count)]
    path.write_bytes(b"".join(doc + b"\n" for doc in docs))


def test_prep_writes_a_loadable_shard_set_whose_manifest_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 20)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--val-fraction",
            "0.2",
            "--seed",
            "0",
            "--shard-tokens",
            "1000",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest_path = output / "manifest.json"
    assert manifest_path.exists()

    manifest = Manifest.load(manifest_path)
    assert manifest.seed == 0
    assert manifest.val_fraction == 0.2
    assert len(manifest.shards) > 0

    train = ShardedTokens.open(manifest_path, tokenizer_path, "train")
    val = ShardedTokens.open(manifest_path, tokenizer_path, "val")
    assert len(train) > 0
    assert len(val) > 0

    total_tokens = sum(entry.tokens for entry in manifest.shards)
    assert total_tokens == len(train) + len(val)
    # Pinned to the exact wording the CLI prints, and to each split
    # individually: a bare `str(total_tokens)` substring can also match
    # inside the printed tokenizer sha256, and asserting only the total
    # cannot catch the two counts being swapped or filtered backwards.
    assert f"{len(train)} train tokens" in result.output
    assert f"{len(val)} val tokens" in result.output
    assert f"{total_tokens} total tokens" in result.output


def test_prep_respects_limit(tmp_path: Path) -> None:
    """`--limit` is what makes an end-to-end CLI test possible without a real
    corpus: only the first N documents are processed, and that limit is
    recorded on the manifest."""
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 20)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--val-fraction",
            "0.0",
            "--seed",
            "0",
            "--shard-tokens",
            "1000",
            "--limit",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = Manifest.load(output / "manifest.json")
    assert manifest.limit == 5


def test_data_group_shows_help_rather_than_erroring() -> None:
    """Pins the exit code, not just the presence of "prep" in the output:
    without it, this test would equally pass if `data` broke and its
    traceback happened to mention the word "prep" somewhere. Click's
    `no_args_is_help=True` exits 2 for a bare group invocation even though
    it is displaying help rather than reporting a usage error -- the same
    code `tokenizer` (with the same setting) exits for the same reason."""
    result = runner.invoke(app, ["data"])
    assert result.exit_code == 2
    assert "prep" in result.output


def test_a_missing_source_file_fails_fast_with_a_named_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing-corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(missing),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    # The adjacent form, not "source" alone: pytest's `tmp_path` basename is
    # the test function name truncated to 30 characters, and this test's
    # name contains "source" too, so a bare `"source" in squashed` passes
    # even if `--source` and `--tokenizer`'s validators were swapped. Only
    # the run-together form pins which option Typer actually named.
    assert "invalidvalueforsource" in squashed
    assert "doesnotexist" in squashed
    assert not output.exists()


def test_a_missing_tokenizer_file_fails_fast_with_a_named_error(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    missing_tokenizer = tmp_path / "missing-tokenizer.json"
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(missing_tokenizer),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefortokenizer" in squashed
    assert "doesnotexist" in squashed
    assert not output.exists()


def test_a_source_path_that_is_an_existing_directory_fails_fast(tmp_path: Path) -> None:
    source_dir = tmp_path / "sourcedir"
    source_dir.mkdir()
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source_dir),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvalueforsource" in squashed
    assert "directory" in squashed
    assert not output.exists()


def test_a_tokenizer_path_that_is_an_existing_directory_fails_fast(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_dir = tmp_path / "tokenizerdir"
    tokenizer_dir.mkdir()
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefortokenizer" in squashed
    assert "directory" in squashed
    assert not output.exists()


def test_an_output_path_that_is_an_existing_file_fails_before_preparing(tmp_path: Path) -> None:
    """Without `dir_okay=False` this would only fail deep inside `prepare`,
    after tokenizing has already run. It must be rejected before any work
    starts, mirroring `tokenizer train`'s `--output` precedent."""
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "not-a-dir"
    output.write_text("existing file")

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvalueforoutput" in squashed


def test_an_output_path_whose_nearest_existing_ancestor_is_unwritable_fails_before_preparing(
    tmp_path: Path,
) -> None:
    """`prepare` reads the whole tokenizer and corpus, computing both their
    sha256 digests, before it ever calls `output_dir.mkdir(...)`. Without a
    dedicated check, an unwritable destination would only surface as a bare
    `PermissionError` traceback after that whole-corpus read -- the same
    defect shape `tokenizer train`'s `--output` review round already fixed
    once, just one layer further in.

    `typer.Option(..., writable=True)` cannot catch this either, since
    click's writability check only runs when the path already exists, and a
    fresh `--output` directory commonly does not.
    """
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    readonly_parent = tmp_path / "readonly"
    readonly_parent.mkdir()
    output = readonly_parent / "shards"

    readonly_parent.chmod(0o500)
    try:
        result = runner.invoke(
            app,
            [
                "data",
                "prep",
                "--source",
                str(source),
                "--tokenizer",
                str(tokenizer_path),
                "--output",
                str(output),
            ],
        )
    finally:
        readonly_parent.chmod(0o700)

    assert result.exit_code == 2, result.output
    squashed = _squash(result.output)
    assert "invalidvalueforoutput" in squashed
    assert "notwritable" in squashed
    assert not output.exists()


def test_a_val_fraction_above_one_is_rejected(tmp_path: Path) -> None:
    """A typo this size must be an argument error, not a run that fails deep
    inside `prepare` after tokenizing has already started."""
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--val-fraction",
            "1.5",
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "valfraction" in squashed
    assert not output.exists()


def test_a_negative_val_fraction_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--val-fraction",
            "-0.1",
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "valfraction" in squashed
    assert not output.exists()


def test_a_zero_shard_tokens_is_rejected(tmp_path: Path) -> None:
    """`ShardWriter` itself rejects a non-positive `shard_tokens`, but that
    check must not be reached only after tokenizing an entire corpus first."""
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--shard-tokens",
            "0",
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "shardtokens" in squashed
    assert not output.exists()


def test_a_zero_max_chunk_bytes_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--max-chunk-bytes",
            "0",
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "maxchunkbytes" in squashed
    assert not output.exists()


def test_a_zero_limit_is_rejected(tmp_path: Path) -> None:
    """`--limit 0` would silently produce an empty, useless shard set rather
    than the "no limit" behaviour `None` gives; it must be an argument error."""
    source = tmp_path / "corpus.txt"
    _write_corpus(source, 5)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
            "--limit",
            "0",
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "limit" in squashed
    assert not output.exists()


def test_an_over_long_chunk_is_rejected_and_writes_nothing_new(tmp_path: Path) -> None:
    """`prepare` itself raises here (design spec section 5): a `ValueError`
    naming the document and the observed length. The CLI's own
    `except ValueError` block -- the only error-handling code this verb adds
    -- must turn that into a clean exit 2 naming both, not let it surface as
    an uncaught traceback (which would also be a nonzero, non-2 exit code,
    so `exit_code != 0` alone cannot tell the two apart).

    `--output` is a directory `prepare` creates as part of its own work, so
    "writes nothing" cannot mean the directory never exists: `prepare`
    creates it before it discovers the over-long document, since a corpus
    is only scanned once, in document order. What matters is that no shard
    file bears a name a later `data prep` run would treat as trustworthy.
    The corpus here has this as its only, first document, so no
    `ShardWriter` ever opens a shard file before the raise: the directory
    is left completely empty, not just manifest-less.
    """
    source = tmp_path / "corpus.txt"
    over_long = b"x" * 2000
    source.write_bytes(over_long + b"\n")
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    output = tmp_path / "shards"

    result = runner.invoke(
        app,
        [
            "data",
            "prep",
            "--source",
            str(source),
            "--tokenizer",
            str(tokenizer_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2, result.output
    squashed = _squash(result.output)
    assert "document0" in squashed
    assert "maxchunkbytes1024" in squashed
    assert not (output / "manifest.json").exists()
    assert list(output.iterdir()) == []
