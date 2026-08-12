"""BPE training.

Two implementations live here. `train_naive` recounts every pair on every
merge; it is short enough to check by eye and exists as the oracle for the
differential test. `train_indexed` is the shipped path.

The reason for keeping both: a stale pair count in an incremental
implementation produces a *worse* merge table, not an invalid one. The
tokenizer still round-trips perfectly while the compression ratio this repo
publishes is quietly wrong. No round-trip test can catch that. A differential
test against an implementation that is correct by inspection can.
"""

from collections import defaultdict
from itertools import pairwise

from nanoscope.tokenizer.pretokenize import pretokenize
from nanoscope.tokenizer.vocab import FIRST_MERGE_ID

Pair = tuple[int, int]
Word = tuple[int, ...]


def _word_counts(corpus: bytes) -> dict[Word, int]:
    """Collapse the corpus to unique pre-token chunks and their frequencies."""
    counts: dict[Word, int] = defaultdict(int)
    for chunk in pretokenize(corpus):
        counts[tuple(chunk)] += 1
    return dict(counts)


def _pair_counts(words: dict[Word, int]) -> dict[Pair, int]:
    """Count every adjacent pair, weighted by how often its word occurs."""
    counts: dict[Pair, int] = defaultdict(int)
    for word, freq in words.items():
        for pair in pairwise(word):
            counts[pair] += freq
    return dict(counts)


def _best_pair(counts: dict[Pair, int]) -> Pair | None:
    """Most frequent pair, ties broken by the lexicographically smallest pair.

    The tie-break is not cosmetic. Without a total order here, `train_naive`
    and `train_indexed` can make different but equally defensible choices, and
    the differential test that justifies having two implementations becomes
    meaningless.
    """
    if not counts:
        return None
    return min((-count, pair) for pair, count in counts.items())[1]


def _merge_word(word: Word, pair: Pair, new_id: int) -> Word:
    """Replace every non-overlapping left-to-right occurrence of `pair`."""
    out: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_naive(corpus: bytes, vocab_size: int) -> list[Pair]:
    """Train by rescanning every pair on every merge. The oracle, not the path.

    Stops early if the corpus runs out of adjacent pairs, so the resulting
    vocabulary can be smaller than `vocab_size` on small inputs.
    """
    n_merges = max(0, vocab_size - FIRST_MERGE_ID)
    words = _word_counts(corpus)
    merges: list[Pair] = []
    for k in range(n_merges):
        best = _best_pair(_pair_counts(words))
        if best is None:
            break
        new_id = FIRST_MERGE_ID + k
        merged: dict[Word, int] = defaultdict(int)
        for word, freq in words.items():
            merged[_merge_word(word, best, new_id)] += freq
        words = dict(merged)
        merges.append(best)
    return merges


def train_indexed(corpus: bytes, vocab_size: int) -> list[Pair]:
    """Train by updating pair counts incrementally. The shipped path.

    Maintains `pair -> set of word indices containing it`, so applying a merge
    only rescans the words that actually held the merged pair rather than the
    whole corpus. Must agree with `train_naive` exactly; the differential test
    in tests/test_train.py enforces that.
    """
    n_merges = max(0, vocab_size - FIRST_MERGE_ID)
    counted = _word_counts(corpus)
    words: list[Word] = list(counted)
    freqs: list[int] = [counted[word] for word in words]

    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)
    for index, word in enumerate(words):
        for pair in pairwise(word):
            pair_counts[pair] += freqs[index]
            where[pair].add(index)

    merges: list[Pair] = []
    for k in range(n_merges):
        best = _best_pair(pair_counts)
        if best is None:
            break
        new_id = FIRST_MERGE_ID + k
        merges.append(best)

        # sorted() materialises the set before iteration, which matters: the
        # loop body mutates `where` entries, including `where[best]` itself.
        for index in sorted(where[best]):
            old = words[index]
            new = _merge_word(old, best, new_id)
            words[index] = new
            freq = freqs[index]
            for pair in pairwise(old):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                where[pair].discard(index)
            for pair in pairwise(new):
                pair_counts[pair] += freq
                where[pair].add(index)

        pair_counts.pop(best, None)
        where.pop(best, None)
    return merges


# The public name. `train_naive` stays module-level but is not re-exported from
# the package: its only caller is the differential test.
train = train_indexed
