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
# Requests FIRST_MERGE_ID + 40 merges, but this corpus only yields 17 before
# the trainer runs out of adjacent pairs (verified by direct measurement, not
# a property this file re-derives): the merge table below is smaller than the
# vocab_size argument suggests.
TRAINED = Tokenizer(train(b"the cat sat on the mat, the cat sat again. " * 20, FIRST_MERGE_ID + 40))
EMPTY = Tokenizer([])

# st.binary() essentially never draws the specific English byte sequences
# TRAINED's merge table was trained on, so a bare `st.binary()` strategy is
# almost entirely byte-identity paths that exercise no merge. Mixing in pieces
# drawn from the training corpus (plus arbitrary short binary noise) gives the
# same total strategy a real chance of hitting merges, without narrowing what
# it covers: `st.one_of(st.binary(), _CORPUS_LIKE)` keeps the original
# unrestricted strategy in the mix.
_PIECES = st.sampled_from(
    [b"the ", b"cat", b" sat", b" mat", b" on", b".", b", ", b" again", b"\xff", b""]
)
_CORPUS_LIKE = st.lists(st.one_of(_PIECES, st.binary(max_size=4)), max_size=20).map(b"".join)


@given(st.one_of(st.binary(), _CORPUS_LIKE))
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


@given(st.one_of(st.binary(), _CORPUS_LIKE))
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


@given(st.one_of(st.binary(), _CORPUS_LIKE))
def test_encode_never_emits_the_special_token(data: bytes) -> None:
    """What keeps the round-trip law total: no input can produce id 256."""
    assert END_OF_TEXT_ID not in TRAINED.encode(data)


def test_the_special_token_decodes_to_its_spelling() -> None:
    assert TRAINED.decode([END_OF_TEXT_ID]) == END_OF_TEXT.encode("utf-8")


def test_decode_rejects_a_negative_id() -> None:
    """Without a bounds check, Python's negative indexing wraps this to the
    highest-rank merge token instead of failing -- the id looks decoded, not
    rejected."""
    with pytest.raises(ValueError, match="-1"):
        TRAINED.decode([-1])


def test_decode_rejects_an_id_past_the_end_of_the_vocabulary() -> None:
    with pytest.raises(ValueError, match=str(TRAINED.vocab_size)):
        TRAINED.decode([TRAINED.vocab_size])


def test_training_actually_compresses() -> None:
    text = b"the cat sat on the mat, the cat sat again. " * 5
    assert len(TRAINED.encode(text)) < len(EMPTY.encode(text))


def test_the_cache_does_not_change_results() -> None:
    """Second call hits the chunk cache; it must return the same ids a fresh,
    cold instance built from the same merges would, not merely the same ids
    as its own first call, which would also pass for a stable-but-wrong cache."""
    data = b"the cat sat on the mat"
    cold = Tokenizer(TRAINED._merges)
    warm = TRAINED
    warm.encode(data)  # populate the cache
    assert warm.encode(data) == cold.encode(data)


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


def test_load_rejects_a_duplicate_merge_pair(tmp_path: Path) -> None:
    """A repeated pair validates its way to a permanently dead vocabulary slot
    at the earlier rank if not rejected here."""
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "pattern": PATTERN_SOURCE,
                "special_tokens": {END_OF_TEXT: END_OF_TEXT_ID},
                "corpus_sha256": None,
                "merges": [[97, 98], [97, 98]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        Tokenizer.load(path)


def test_load_rejects_a_merge_that_references_the_special_token_id(tmp_path: Path) -> None:
    """id 256 is <|endoftext|>, not a byte or an earlier merge; a merge that
    references it is inert but no legitimate file contains one."""
    path = tmp_path / "special-token-ref.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "pattern": PATTERN_SOURCE,
                "special_tokens": {END_OF_TEXT: END_OF_TEXT_ID},
                "corpus_sha256": None,
                "merges": [[97, 256]],
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
    with pytest.raises(ValueError, match="different split pattern"):
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


@given(st.binary(max_size=512))
@settings(deadline=None)
def test_encode_chunks_agrees_with_encode_on_the_same_bytes(data: bytes) -> None:
    """`encode(data)` is defined as `encode_chunks(pretokenize(data))`, and
    `data.prepare` relies on that identity: it measures each document's chunk
    lengths against the quadratic-encode ceiling and then encodes the chunks
    it already has, rather than paying for the regex pass a second time.

    Property-based rather than a fixed example, since the whole value of the
    shortcut is that it holds for arbitrary corpus bytes. A tokenizer with
    real merges, not the identity vocabulary, so chunk boundaries actually
    matter to the result.
    """
    tokenizer = Tokenizer(train(b"the theatre theme, the other one" * 4, 300))
    assert tokenizer.encode_chunks(pretokenize(data)) == tokenizer.encode(data)
