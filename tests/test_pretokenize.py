"""Pre-tokenization tests.

The two property tests here are load-bearing for the whole tokenizer. Total
coverage of the input is what makes the round-trip law in test_tokenizer.py
total, and idempotence is what makes encoding chunk-local.
"""

from hypothesis import given
from hypothesis import strategies as st

from nanoscope.tokenizer.pretokenize import pretokenize


@given(st.binary())
def test_chunks_rejoin_to_the_original_bytes(data: bytes) -> None:
    """The split partitions the input: nothing is dropped, added, or reordered."""
    assert b"".join(pretokenize(data)) == data


@given(st.binary())
def test_every_chunk_resplits_to_itself(data: bytes) -> None:
    """Pre-tokenization is idempotent.

    Encoding is chunk-local, so `encode` re-runs this split on data that has
    already been split. If the pattern does not reproduce its own output, that
    is a real fact about the pattern, not a bug in the encoder.
    """
    for chunk in pretokenize(data):
        assert pretokenize(chunk) == [chunk]


def test_splits_a_sentence_into_leading_space_words() -> None:
    assert pretokenize(b"Hello world!") == [b"Hello", b" world", b"!"]


def test_digit_runs_are_capped_at_three() -> None:
    assert pretokenize(b"12345") == [b"123", b"45"]


def test_invalid_utf8_survives_the_split() -> None:
    """Bytes that are not valid UTF-8 must pass through, not raise or mangle."""
    assert b"".join(pretokenize(b"\xff\xfe hi")) == b"\xff\xfe hi"


def test_empty_input_gives_no_chunks() -> None:
    assert pretokenize(b"") == []
