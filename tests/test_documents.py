"""Tests for document iteration: the seam between a corpus file and the
tokenizer.

Design spec section 4: `encode` is chunk-local and a document boundary is
always a chunk boundary, because `\\s*[\\r\\n]+` is one of the split pattern's
alternatives. That is what makes a newline a safe place to cut a corpus, and
why nothing here may split at an arbitrary byte offset instead.
"""

import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from nanoscope.data.documents import iter_documents

# Documents free of embedded newlines, so each one occupies exactly one line
# and survives a line-oriented round trip. `st.binary()` on its own draws
# newline bytes too, and a document containing one cannot be told apart from
# a separator once written to a line-oriented file, so the strategy is
# narrowed here rather than the property weakened. See the property test
# below for the full round-trip law this buys.
_line_safe_bytes = st.binary().filter(lambda b: b"\n" not in b)


def test_an_empty_file_yields_no_documents(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"")
    assert list(iter_documents(path)) == []


def test_a_missing_trailing_newline_still_yields_the_last_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"first\nsecond")
    assert list(iter_documents(path)) == [b"first", b"second"]


def test_a_blank_line_yields_an_empty_document_rather_than_being_skipped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"first\n\nlast\n")
    assert list(iter_documents(path)) == [b"first", b"", b"last"]


def test_a_trailing_newline_does_not_manufacture_a_phantom_final_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"only\n")
    assert list(iter_documents(path)) == [b"only"]


@given(st.lists(_line_safe_bytes))
def test_documents_terminated_by_newlines_round_trip_exactly(docs: list[bytes]) -> None:
    """Each document is written with its own trailing newline, so every line
    in the file has a terminator and none is a partial final line. With no
    embedded newlines and no unterminated last line, there is no ambiguity
    left for the reader to resolve, so it must recover exactly `docs`.

    Built with `tempfile` rather than the `tmp_path` fixture: `tmp_path` is
    function-scoped and Hypothesis reruns this body many times per test
    function, so a fixture-backed path would alias across examples.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "corpus.txt"
        path.write_bytes(b"".join(doc + b"\n" for doc in docs))
        assert list(iter_documents(path)) == docs
