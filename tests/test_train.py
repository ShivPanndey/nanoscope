"""Trainer tests.

The differential test against the naive oracle arrives in the next task; this
file starts with the cases that pin the naive implementation's behaviour, since
it is about to become the thing that certifies the shipped path.
"""

from nanoscope.tokenizer.train import train_naive
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
