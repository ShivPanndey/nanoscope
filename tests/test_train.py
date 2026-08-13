"""Trainer tests.

Starts with cases that pin `train_naive`'s behaviour by hand-computed example,
then the differential test that checks `train_indexed` against it, then the
one-line pin that `train` is `train_indexed`.

The hand-computed cases below call `train_naive` directly, not `train`. Spec
section 7 says these check `train`'s output; that holds transitively through
`test_indexed_trainer_matches_the_naive_oracle` (which certifies
`train_indexed` against `train_naive` on generated corpora) plus
`test_train_is_the_indexed_implementation` (which pins `train is
train_indexed`), so `train_naive`'s pinned behaviour is `train`'s behaviour by
composition of those two facts.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from nanoscope.tokenizer.train import train, train_indexed, train_naive
from nanoscope.tokenizer.vocab import FIRST_MERGE_ID


def test_no_merges_when_vocab_leaves_no_room() -> None:
    assert train_naive(b"abab abab", FIRST_MERGE_ID) == []


def test_learns_the_most_frequent_pair_first() -> None:
    """`ab` occurs four times, more than any other adjacent pair."""
    merges = train_naive(b"abab abab", FIRST_MERGE_ID + 1)
    assert merges == [(ord("a"), ord("b"))]


def test_second_merge_can_reference_the_first() -> None:
    """After (a,b) becomes 257, the pair (257,257) is what `abab` now contains."""
    merges = train_naive(b"abab abab", FIRST_MERGE_ID + 2)
    assert merges == [(ord("a"), ord("b")), (FIRST_MERGE_ID, FIRST_MERGE_ID)]


def test_ties_go_to_the_lexicographically_smallest_pair() -> None:
    """Every adjacent pair here occurs exactly once, so the tie-break decides.

    Chunks are `ab`, ` cd`. Candidate pairs and their counts:
    (97,98)=1, (32,99)=1, (99,100)=1. The smallest is (32,99).
    """
    merges = train_naive(b"ab cd", FIRST_MERGE_ID + 1)
    assert merges == [(ord(" "), ord("c"))]


def test_merges_never_cross_a_pre_token_boundary() -> None:
    """`b` and `c` are adjacent in the raw bytes but sit in different chunks."""
    merges = train_naive(b"ab cd ab cd ab cd", FIRST_MERGE_ID + 10)
    assert (ord("b"), ord(" ")) not in merges
    assert (ord("b"), ord("c")) not in merges


def test_stops_early_when_no_pairs_remain() -> None:
    """A single-byte corpus has no adjacent pairs at all."""
    assert train_naive(b"a", FIRST_MERGE_ID + 50) == []


@st.composite
def corpora(draw: st.DrawFn) -> bytes:
    """Small byte strings drawn from a deliberately narrow alphabet.

    A narrow alphabet forces collisions and repeated pairs, which is where
    incremental count maintenance actually breaks. Uniform random bytes would
    mostly produce unique pairs and exercise nothing.
    """
    alphabet = draw(st.sampled_from([b"ab", b"abc ", b"ab c\n", b"a1 .\xff"]))
    length = draw(st.integers(min_value=0, max_value=80))
    return bytes(draw(st.lists(st.sampled_from(alphabet), min_size=length, max_size=length)))


@given(corpus=corpora(), extra=st.integers(min_value=0, max_value=60))
@settings(max_examples=200, deadline=None)
def test_indexed_trainer_matches_the_naive_oracle(corpus: bytes, extra: int) -> None:
    """The whole reason two trainers exist.

    A stale pair count in the indexed trainer yields a worse merge table that
    still round-trips perfectly, so this is the only test that can catch it.
    """
    vocab_size = FIRST_MERGE_ID + extra
    assert train_indexed(corpus, vocab_size) == train_naive(corpus, vocab_size)


def test_train_is_the_indexed_implementation() -> None:
    assert train is train_indexed
