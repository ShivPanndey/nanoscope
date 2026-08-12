"""End-to-end test of the `tokenizer train` verb."""

from pathlib import Path

from typer.testing import CliRunner

from nanoscope.cli import app
from nanoscope.tokenizer import Tokenizer
from nanoscope.tokenizer.vocab import FIRST_MERGE_ID

runner = CliRunner()


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
