"""Tokenizer round-trip and encoding tests.

Property 1 is the one that matters: it quantifies over `st.binary()` with no
exclusions, so there is no class of input quietly carved out of the guarantee.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from nanoscope.tokenizer import Tokenizer, train
from nanoscope.tokenizer.pretokenize import pretokenize
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
