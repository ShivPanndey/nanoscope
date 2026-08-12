"""Tokenizer round-trip and encoding tests.

Property 1 is the one that matters: it quantifies over `st.binary()` with no
exclusions, so there is no class of input quietly carved out of the guarantee.
"""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from nanoscope.tokenizer import Tokenizer, train
from nanoscope.tokenizer.pretokenize import PATTERN_SOURCE, pretokenize
from nanoscope.tokenizer.vocab import END_OF_TEXT, END_OF_TEXT_ID, FIRST_MERGE_ID

# One shared trained tokenizer: training is the slow part, and every property
# below holds for any merge table, so retraining per example buys nothing.
TRAINED = Tokenizer(train(b"the cat sat on the mat, the cat sat again. " * 20, FIRST_MERGE_ID + 40))
EMPTY = Tokenizer([])


@given(st.binary())
@settings(max_examples=300)
def test_bytes_round_trip_exactly(data: bytes) -> None:
    assert TRAINED.decode(TRAINED.encode(data)) == data


# st.text()'s default alphabet is UTF-8-encodable only, which
# excludes the very surrogates this test is named for.
@given(st.text(alphabet=st.characters()))
@settings(max_examples=300)
def test_text_round_trips_including_lone_surrogates(text: str) -> None:
    assert TRAINED.decode_str(TRAINED.encode_str(text)) == text


def test_a_lone_surrogate_round_trips() -> None:
    """Named regression case for the input that breaks strict-UTF-8 designs."""
    assert TRAINED.decode_str(TRAINED.encode_str("\ud800")) == "\ud800"


def test_emoji_and_combining_characters_round_trip() -> None:
    text = "éclair \U0001f9d1‍\U0001f680 नमस्ते"
    assert TRAINED.decode_str(TRAINED.encode_str(text)) == text


@given(st.binary())
def test_every_id_is_within_the_vocabulary(data: bytes) -> None:
    assert all(0 <= i < TRAINED.vocab_size for i in TRAINED.encode(data))


@given(st.binary())
def test_encoding_is_chunk_local(data: bytes) -> None:
    """Merges never cross a pre-token boundary."""
    per_chunk = [i for chunk in pretokenize(data) for i in TRAINED.encode(chunk)]
    assert TRAINED.encode(data) == per_chunk


@given(st.binary())
def test_an_untrained_tokenizer_emits_raw_byte_values(data: bytes) -> None:
    """Pins the id-equals-byte-value layout invariant."""
    assert EMPTY.encode(data) == list(data)


@given(st.binary())
def test_encode_never_emits_the_special_token(data: bytes) -> None:
    """What keeps the round-trip law total: no input can produce id 256."""
    assert END_OF_TEXT_ID not in TRAINED.encode(data)


def test_the_special_token_decodes_to_its_spelling() -> None:
    assert TRAINED.decode([END_OF_TEXT_ID]) == END_OF_TEXT.encode("utf-8")


def test_training_actually_compresses() -> None:
    text = b"the cat sat on the mat, the cat sat again. " * 5
    assert len(TRAINED.encode(text)) < len(EMPTY.encode(text))


def test_the_cache_does_not_change_results() -> None:
    """Second call hits the chunk cache; it must return the same ids."""
    data = b"the cat sat on the mat"
    assert TRAINED.encode(data) == TRAINED.encode(data)


def test_a_saved_tokenizer_loads_back_identically(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    TRAINED.save(path)
    loaded = Tokenizer.load(path)
    data = b"the cat sat on the mat \xff\x00"
    assert loaded.encode(data) == TRAINED.encode(data)
    assert loaded.vocab_size == TRAINED.vocab_size


def test_the_corpus_hash_survives_a_round_trip(tmp_path: Path) -> None:
    """Published compression numbers have to trace to the exact training bytes."""
    path = tmp_path / "tokenizer.json"
    Tokenizer([], corpus_sha256="deadbeef").save(path)
    assert Tokenizer.load(path).corpus_sha256 == "deadbeef"


def test_load_rejects_a_merge_that_references_an_undefined_id(tmp_path: Path) -> None:
    """The first merge may only reference ids below 257; 300 does not exist yet."""
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "pattern": PATTERN_SOURCE,
                "special_tokens": {END_OF_TEXT: END_OF_TEXT_ID},
                "corpus_sha256": None,
                "merges": [[97, 300]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        Tokenizer.load(path)


def test_load_rejects_a_file_written_with_a_different_split_pattern(tmp_path: Path) -> None:
    """A merge table trained under another splitter is not usable with this one."""
    path = tmp_path / "foreign.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "pattern": r"\w+|\s+",
                "special_tokens": {END_OF_TEXT: END_OF_TEXT_ID},
                "corpus_sha256": None,
                "merges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pattern"):
        Tokenizer.load(path)


def test_load_rejects_a_file_with_an_unknown_field(tmp_path: Path) -> None:
    """A misspelled field (e.g. corpus_sha_256) must not be silently dropped.

    With extra fields forbidden, a typo'd key fails loudly instead of loading
    as if the field it was meant to be were simply absent.
    """
    path = tmp_path / "typo.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "pattern": PATTERN_SOURCE,
                "special_tokens": {END_OF_TEXT: END_OF_TEXT_ID},
                "corpus_sha_256": "deadbeef",
                "merges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        Tokenizer.load(path)


def test_load_rejects_a_file_with_an_unsupported_version(tmp_path: Path) -> None:
    """A version this build does not know how to read must fail, not load partially."""
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "pattern": PATTERN_SOURCE,
                "special_tokens": {END_OF_TEXT: END_OF_TEXT_ID},
                "corpus_sha256": None,
                "merges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        Tokenizer.load(path)


def test_load_rejects_a_file_with_mismatched_special_tokens(tmp_path: Path) -> None:
    """The special-token map is only meaningful alongside the id `decode` hardcodes."""
    path = tmp_path / "mismatched.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "pattern": PATTERN_SOURCE,
                "special_tokens": {END_OF_TEXT: 999},
                "corpus_sha256": None,
                "merges": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="special token"):
        Tokenizer.load(path)
