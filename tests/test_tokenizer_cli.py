"""End-to-end test of the `tokenizer train` verb."""

from pathlib import Path

from typer.testing import CliRunner

from nanoscope.cli import app
from nanoscope.tokenizer import Tokenizer
from nanoscope.tokenizer.vocab import FIRST_MERGE_ID

runner = CliRunner()


def _squash(text: str) -> str:
    """Collapse a Rich-rendered error panel into one lowercase alnum-only string.

    CliRunner's default 80-column width word-wraps long paths and messages,
    sometimes mid-word, with box-drawing borders spliced into the break. Pytest's
    `tmp_path` is deeply nested and long enough to trigger this reliably, so
    comparing raw substrings against `result.output` is flaky. Stripping
    whitespace, punctuation, and border characters makes the assertion
    independent of exactly where Rich chose to wrap.
    """
    return "".join(ch for ch in text if ch.isalnum()).lower()


def test_train_writes_a_loadable_tokenizer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"the cat sat on the mat. " * 50)
    out = tmp_path / "tokenizer.json"

    result = runner.invoke(
        app,
        [
            "tokenizer",
            "train",
            "--input",
            str(corpus),
            "--output",
            str(out),
            "--vocab-size",
            str(FIRST_MERGE_ID + 20),
        ],
    )

    assert result.exit_code == 0, result.output
    loaded = Tokenizer.load(out)
    # Not an equality check: this corpus has only eight distinct pre-token
    # chunks, so the trainer runs out of adjacent pairs and stops early. The
    # exact merge count is a property of the corpus, not of correctness.
    assert FIRST_MERGE_ID < loaded.vocab_size <= FIRST_MERGE_ID + 20
    assert loaded.decode(loaded.encode(b"the cat")) == b"the cat"


def test_train_records_the_corpus_hash(tmp_path: Path) -> None:
    """The hash printed to stdout must match the one stored in the artifact."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"the cat sat on the mat. " * 50)
    out = tmp_path / "tokenizer.json"

    result = runner.invoke(
        app, ["tokenizer", "train", "--input", str(corpus), "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    digest = Tokenizer.load(out).corpus_sha256
    assert digest is not None
    assert digest in result.output


def test_tokenizer_group_shows_help_rather_than_erroring() -> None:
    result = runner.invoke(app, ["tokenizer"])
    assert "train" in result.output


def test_a_missing_input_file_fails_fast_with_a_named_error(tmp_path: Path) -> None:
    """No traceback: Typer's own Path validation rejects it before any work starts."""
    missing = tmp_path / "missing-corpus.txt"
    out = tmp_path / "tokenizer.json"

    result = runner.invoke(
        app, ["tokenizer", "train", "--input", str(missing), "--output", str(out)]
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "input" in squashed
    assert "doesnotexist" in squashed
    assert not out.exists()


def test_a_missing_output_directory_fails_before_training(tmp_path: Path) -> None:
    """The typo is knowable before the first pair is counted, so it must not train
    to completion first only to fail on the write."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"the cat sat on the mat. " * 50)
    out = tmp_path / "no" / "such" / "dir" / "tokenizer.json"

    result = runner.invoke(
        app, ["tokenizer", "train", "--input", str(corpus), "--output", str(out)]
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "output" in squashed
    assert "doesnotexist" in squashed
    assert not out.parent.exists()


def test_an_output_path_that_is_an_existing_directory_fails_before_training(
    tmp_path: Path,
) -> None:
    """Without `dir_okay=False` this only fails inside `save()`, after a full
    training run is spent. Typer must reject it before any work starts."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"the cat sat on the mat. " * 50)
    out = tmp_path / "adir"
    out.mkdir()

    result = runner.invoke(
        app, ["tokenizer", "train", "--input", str(corpus), "--output", str(out)]
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "output" in squashed
    assert "directory" in squashed


def test_an_input_path_that_is_an_existing_directory_fails_fast(tmp_path: Path) -> None:
    """Mirrors the --output case above: Typer's own Path validation rejects a
    directory before any work starts."""
    input_dir = tmp_path / "inputdir"
    input_dir.mkdir()
    out = tmp_path / "tokenizer.json"

    result = runner.invoke(
        app, ["tokenizer", "train", "--input", str(input_dir), "--output", str(out)]
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "input" in squashed
    assert "directory" in squashed
    assert not out.exists()


def test_a_vocab_size_at_or_below_first_merge_id_is_rejected(tmp_path: Path) -> None:
    """--vocab-size 5 used to silently produce a 0-merge, vocab-257 tokenizer at
    exit 0. A typo this small must be an argument error, not a no-op success."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"the cat sat on the mat. " * 50)
    out = tmp_path / "tokenizer.json"

    result = runner.invoke(
        app,
        [
            "tokenizer",
            "train",
            "--input",
            str(corpus),
            "--output",
            str(out),
            "--vocab-size",
            "5",
        ],
    )

    assert result.exit_code == 2
    squashed = _squash(result.output)
    assert "invalidvaluefor" in squashed
    assert "vocabsize" in squashed
    assert not out.exists()
