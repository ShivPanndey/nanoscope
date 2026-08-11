# Byte-level BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a corpus-agnostic byte-level BPE tokenizer — library, CLI verb, and property test suite — implementing `docs/specs/2026-08-11-tokenizer-design.md`.

**Architecture:** `bytes` is the fundamental domain; `str` support is a thin `surrogatepass` wrapper. Pre-tokenization uses tiktoken's cl100k_base split pattern, bridged from bytes to str with `surrogateescape`. Two trainers are written — a naive oracle and an indexed fast path — and checked against each other by a Hypothesis differential test, because an incremental pair-count bug degrades the merge table without breaking any round-trip property.

**Tech Stack:** Python 3.13, `regex` (for `\p{L}` / `\p{N}`), Pydantic v2 (serialization), Hypothesis (property tests), Typer (CLI), uv, ruff, mypy strict, pytest.

## Global Constraints

- Python `>=3.13`. Run everything through `uv run`.
- `make check` must be green before any task is considered done: ruff, `ruff format --check`, mypy strict, pytest.
- ruff line-length is 100. The `ANN` ruleset is on, so **every function needs full annotations, including `-> None` on tests**. `tests/**` waives `ANN001` (test parameters) only.
- `B905` is on: every `zip(...)` call must pass `strict=` explicitly. This plan uses `strict=False` throughout, deliberately — pairing a sequence with its own tail always has one fewer element and `strict=True` would raise.
- mypy is `strict` with `warn_unreachable` and `disallow_any_generics`. No bare `dict`/`list` annotations.
- **Commits are authored by Shivank Pandey alone. Never add a `Co-Authored-By` trailer.**
- Never run git write commands from `/Users/shivpanndey05` or any uninitialised subdirectory. All git commands in this plan run from the repo root, `/Users/shivpanndey05/dev/portfolio/nanoscope`.
- No fabricated numbers anywhere, in code comments, docs, or commit messages. Trainer wall-clock is measured when the real training run happens, not estimated.
- Vocabulary layout is fixed: ids `0..255` are raw bytes with `id == byte value`, id `256` is `<|endoftext|>`, ids `257..8191` are learned merges in rank order.
- Tie-break for equal-frequency pairs is fixed and shared by both trainers: `min((-count, pair) for pair, count in counts.items())[1]`. Highest count wins; ties go to the lexicographically smallest pair.
- Tests live flat in `tests/`, matching the existing `tests/test_cli.py`. Do not create `tests/` subpackages.

---

## File Structure

**Deviation from spec §3, recorded here deliberately.** The spec lists four files. This plan adds a fifth, `vocab.py`, because the trainer must assign ids to new merges starting at `FIRST_MERGE_ID` so that later merges can reference earlier ones directly — so `train.py` needs the layout constants, while spec §3 forbids `train.py` importing `tokenizer.py`. The alternative, having the trainer emit rank-relative ids that `tokenizer.py` offsets on load, is exactly the index arithmetic that produces silently-wrong vocabulary tables. `vocab.py` also houses the Pydantic serialization model, which both `save` and `load` need.

| File | Responsibility |
|---|---|
| `src/nanoscope/tokenizer/__init__.py` | Public exports: `Tokenizer`, `train` |
| `src/nanoscope/tokenizer/vocab.py` | Layout constants and the `TokenizerFile` Pydantic model |
| `src/nanoscope/tokenizer/pretokenize.py` | The cl100k split pattern and `pretokenize(bytes) -> list[bytes]` |
| `src/nanoscope/tokenizer/train.py` | `train_naive` (oracle), `train_indexed` (shipped), `train` (alias) |
| `src/nanoscope/tokenizer/tokenizer.py` | `Tokenizer`: encode/decode, str wrappers, save/load |
| `src/nanoscope/cli.py` | *modify* — add the `tokenizer train` sub-app |
| `pyproject.toml` | *modify* — add the `regex` runtime dependency |
| `tests/test_pretokenize.py` | Split coverage, idempotence, known splits |
| `tests/test_train.py` | Hand-computed merges, tie-break, differential test |
| `tests/test_tokenizer.py` | Round-trip properties, save/load, integrity rule |
| `tests/test_tokenizer_cli.py` | The `tokenizer train` verb end to end |
| `docs/DECISIONS.md` | *modify* — ADR-0004 |
| `NEXT.md` (repo root) | *modify* — hand off to the data pipeline |

---

## Task 1: Pre-tokenizer

**Files:**
- Modify: `pyproject.toml` (add `regex` dependency, add mypy override)
- Create: `src/nanoscope/tokenizer/__init__.py`
- Create: `src/nanoscope/tokenizer/pretokenize.py`
- Test: `tests/test_pretokenize.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `pretokenize(data: bytes) -> list[bytes]` and the module constant `PATTERN_SOURCE: str`, both from `nanoscope.tokenizer.pretokenize`. Task 2 uses `pretokenize`; Task 5 uses `PATTERN_SOURCE`.

- [ ] **Step 1: Add the `regex` dependency**

In `pyproject.toml`, add `regex` to `[project].dependencies`:

```toml
dependencies = [
    "torch>=2.6",
    "numpy>=2.1",
    "pydantic>=2.9",
    "typer>=0.15",
    "regex>=2024.11",
]
```

Then append a mypy override next to the existing `torch` one:

```toml
# regex does not ship type stubs.
[[tool.mypy.overrides]]
module = ["regex.*"]
ignore_missing_imports = true
```

Run: `uv sync --all-groups`

- [ ] **Step 2: Create the package `__init__.py` as a stub**

Create `src/nanoscope/tokenizer/__init__.py`:

```python
"""Byte-level BPE tokenizer."""
```

It stays a stub until Task 4, when there is something worth exporting.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_pretokenize.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pretokenize.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'nanoscope.tokenizer.pretokenize'`

- [ ] **Step 5: Implement the pre-tokenizer**

Create `src/nanoscope/tokenizer/pretokenize.py`:

```python
"""Pre-tokenization: split bytes into chunks that BPE merges never cross.

The split pattern is cl100k_base's, taken verbatim from tiktoken (MIT licence).
Using tiktoken's exact pattern rather than an approximation is deliberate. The
compression comparison promised in DESIGN.md section 6 is only a clean read on
this repo's merge table if both sides split text the same way; with different
splitters it would report the combined effect of two splitters and two merge
tables.
"""

import regex

# tiktoken cl100k_base, MIT licence. Needs \p{L} and \p{N}, which the standard
# library `re` does not support -- hence the `regex` dependency.
PATTERN_SOURCE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_PATTERN = regex.compile(PATTERN_SOURCE)


def pretokenize(data: bytes) -> list[bytes]:
    """Split `data` into pre-token chunks that rejoin to exactly `data`.

    The pattern is defined over `str` while the tokenizer's domain is `bytes`.
    `surrogateescape` bridges the two: it is total over all byte strings and
    byte-exact on re-encode, so no input is rejected and none is altered.

    `finditer` rather than `findall`: the pattern contains an inline group, and
    `findall` would return group contents instead of whole matches.
    """
    text = data.decode("utf-8", errors="surrogateescape")
    return [m.group().encode("utf-8", errors="surrogateescape") for m in _PATTERN.finditer(text)]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pretokenize.py -q`
Expected: 6 passed.

If `test_every_chunk_resplits_to_itself` fails, **stop and report the counterexample** rather than weakening the test. Idempotence is an assumption the encoder design rests on; a failure changes the design, not the test.

- [ ] **Step 7: Run the full check**

Run: `make check`
Expected: ruff, ruff format, mypy, and pytest all clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/nanoscope/tokenizer/ tests/test_pretokenize.py
git commit -m "feat: add pre-tokenizer with tiktoken's cl100k split pattern

Bytes are bridged to the str-level pattern with surrogateescape, which is
total over all byte strings and byte-exact on re-encode. Two property
tests carry the design: the chunks rejoin to exactly the input, and every
chunk re-splits to itself. The second is what lets the encoder treat
merging as chunk-local."
```

---

## Task 2: Vocabulary layout and the naive trainer

**Files:**
- Create: `src/nanoscope/tokenizer/vocab.py`
- Create: `src/nanoscope/tokenizer/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `pretokenize(data: bytes) -> list[bytes]` from Task 1.
- Produces, from `nanoscope.tokenizer.vocab`: `BYTE_TOKENS: int = 256`, `END_OF_TEXT: str`, `END_OF_TEXT_ID: int = 256`, `FIRST_MERGE_ID: int = 257`, `DEFAULT_VOCAB_SIZE: int = 8192`.
- Produces, from `nanoscope.tokenizer.train`: type aliases `Pair = tuple[int, int]` and `Word = tuple[int, ...]`; `train_naive(corpus: bytes, vocab_size: int) -> list[Pair]`; and the private helpers `_word_counts`, `_pair_counts`, `_best_pair`, `_merge_word` that Task 3 reuses.

- [ ] **Step 1: Write the vocabulary layout module**

Create `src/nanoscope/tokenizer/vocab.py`:

```python
"""Vocabulary layout, shared by the trainer and the tokenizer.

Its own module because both `train.py` and `tokenizer.py` need these constants
while `train.py` must not import `tokenizer.py`: keeping that dependency
one-way means the trainer's differential test exercises the trainer alone,
with no encoder in the path to muddy a failure.

Layout:

    ids    0..255   raw bytes, id == byte value
    id       256    <|endoftext|>
    ids 257..8191   learned merges, in rank order

The id-equals-byte-value invariant is what makes encoder bugs visible by
inspection rather than only by test failure.
"""

BYTE_TOKENS = 256
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_ID = 256
FIRST_MERGE_ID = 257
DEFAULT_VOCAB_SIZE = 8192
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_train.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'nanoscope.tokenizer.train'`

- [ ] **Step 4: Implement the naive trainer**

Create `src/nanoscope/tokenizer/train.py`:

```python
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
        for pair in zip(word, word[1:], strict=False):
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train.py -q`
Expected: 6 passed.

- [ ] **Step 6: Run the full check**

Run: `make check`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/nanoscope/tokenizer/vocab.py src/nanoscope/tokenizer/train.py tests/test_train.py
git commit -m "feat: add vocabulary layout and the naive BPE trainer

Layout constants live in their own module so the trainer can assign merge
ids without importing the tokenizer, keeping that dependency one-way.

The naive trainer rescans every pair on every merge. It is not the shipped
path; it is the oracle the indexed trainer will be checked against. The
equal-frequency tie-break is fixed here for the same reason: without a
total order the two implementations could diverge legitimately."
```

---

## Task 3: Indexed trainer and the differential test

**Files:**
- Modify: `src/nanoscope/tokenizer/train.py` (append `train_indexed` and `train`)
- Test: `tests/test_train.py` (append the differential test)

**Interfaces:**
- Consumes: `Pair`, `Word`, `_word_counts`, `_pair_counts`, `_best_pair`, `_merge_word`, `train_naive` from Task 2; `FIRST_MERGE_ID` from `vocab`.
- Produces: `train_indexed(corpus: bytes, vocab_size: int) -> list[Pair]` and `train`, a module-level alias bound to `train_indexed`. Tasks 5 and 6 call `train`.

- [ ] **Step 1: Write the failing differential test**

Append to `tests/test_train.py`:

```python
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
```

Update the imports at the top of `tests/test_train.py` to:

```python
from hypothesis import given, settings
from hypothesis import strategies as st

from nanoscope.tokenizer.train import train, train_indexed, train_naive
from nanoscope.tokenizer.vocab import FIRST_MERGE_ID
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train.py -q`
Expected: collection error, `ImportError: cannot import name 'train_indexed'`

- [ ] **Step 3: Implement the indexed trainer**

Append to `src/nanoscope/tokenizer/train.py`:

```python
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
        for pair in zip(word, word[1:], strict=False):
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
            for pair in zip(old, old[1:], strict=False):
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                where[pair].discard(index)
            for pair in zip(new, new[1:], strict=False):
                pair_counts[pair] += freq
                where[pair].add(index)

        pair_counts.pop(best, None)
        where.pop(best, None)
    return merges


# The public name. `train_naive` stays module-level but is not re-exported from
# the package: its only caller is the differential test.
train = train_indexed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train.py -q`
Expected: 8 passed.

If the differential test fails, Hypothesis prints a minimal counterexample. Debug the **indexed** implementation against it — `train_naive` is the reference and is not to be adjusted to match.

- [ ] **Step 5: Run the full check**

Run: `make check`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/nanoscope/tokenizer/train.py tests/test_train.py
git commit -m "feat: add indexed BPE trainer, certified against the naive oracle

Maintains pair -> word-index sets so each merge rescans only the words it
touched. Hypothesis checks it against the naive trainer over corpora drawn
from narrow alphabets, which force the repeated and overlapping pairs where
incremental count maintenance actually breaks."
```

---

## Task 4: Encoding and decoding

**Files:**
- Create: `src/nanoscope/tokenizer/tokenizer.py`
- Modify: `src/nanoscope/tokenizer/__init__.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: `pretokenize` from Task 1; `train`, `Pair` from Tasks 2-3; `BYTE_TOKENS`, `END_OF_TEXT`, `END_OF_TEXT_ID`, `FIRST_MERGE_ID` from `vocab`.
- Produces: `Tokenizer(merges: list[Pair], corpus_sha256: str | None = None)` with methods `encode(data: bytes) -> list[int]`, `decode(ids: list[int]) -> bytes`, `encode_str(text: str) -> list[int]`, `decode_str(ids: list[int]) -> str`, and the property `vocab_size: int`. Exported from `nanoscope.tokenizer` alongside `train`. Task 5 adds `save`/`load` to this same class.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tokenizer.py`:

```python
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


@given(st.text())
@settings(max_examples=300)
def test_text_round_trips_including_lone_surrogates(text: str) -> None:
    assert TRAINED.decode_str(TRAINED.encode_str(text)) == text


def test_a_lone_surrogate_round_trips() -> None:
    """Named regression case for the input that breaks strict-UTF-8 designs."""
    assert TRAINED.decode_str(TRAINED.encode_str("\ud800")) == "\ud800"


def test_emoji_and_combining_characters_round_trip() -> None:
    text = "éclair \U0001f9d1‍\U0001f680 नमस्ते"
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: collection error, `ImportError: cannot import name 'Tokenizer'`

- [ ] **Step 3: Implement the tokenizer**

Create `src/nanoscope/tokenizer/tokenizer.py`:

```python
"""The tokenizer itself: encoding, decoding, and the str convenience layer."""

from nanoscope.tokenizer.pretokenize import pretokenize
from nanoscope.tokenizer.train import Pair
from nanoscope.tokenizer.vocab import BYTE_TOKENS, END_OF_TEXT, FIRST_MERGE_ID

# Arbitrary binary input has unbounded chunk diversity, so the chunk cache
# needs a ceiling. Clearing wholesale rather than evicting least-recently-used
# keeps this a plain dict: real corpora have few enough unique chunks that the
# limit is never reached, so eviction policy would be untested code.
CACHE_LIMIT = 100_000


def _build_vocab(merges: list[Pair]) -> list[bytes]:
    """Materialise id -> bytes. Merges are in rank order, so this is one pass."""
    vocab = [bytes([i]) for i in range(BYTE_TOKENS)]
    vocab.append(END_OF_TEXT.encode("utf-8"))
    for first, second in merges:
        vocab.append(vocab[first] + vocab[second])
    return vocab


class Tokenizer:
    """Byte-level BPE tokenizer.

    `bytes` is the fundamental domain. `decode(encode(data)) == data` holds for
    every byte string with no exceptions, which is only true because `encode`
    never emits the special token: nothing in the input can map to id 256.

    The `str` methods are a thin wrapper using `surrogatepass`, which is total
    on `str`, so lone surrogates round-trip rather than raising.
    """

    def __init__(self, merges: list[Pair], corpus_sha256: str | None = None) -> None:
        self._merges: list[Pair] = list(merges)
        self._ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self._merges)}
        self._vocab: list[bytes] = _build_vocab(self._merges)
        self.corpus_sha256 = corpus_sha256
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, data: bytes) -> list[int]:
        ids: list[int] = []
        for chunk in pretokenize(data):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return list(cached)

        ids = list(chunk)
        while len(ids) >= 2:
            best_rank: int | None = None
            best_index = 0
            for index, pair in enumerate(zip(ids, ids[1:], strict=False)):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_rank is None:
                break
            ids[best_index : best_index + 2] = [FIRST_MERGE_ID + best_rank]

        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        self._cache[chunk] = list(ids)
        return ids

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self._vocab[i] for i in ids)

    def encode_str(self, text: str) -> list[int]:
        return self.encode(text.encode("utf-8", errors="surrogatepass"))

    def decode_str(self, ids: list[int]) -> str:
        return self.decode(ids).decode("utf-8", errors="surrogatepass")
```

- [ ] **Step 4: Export the public names**

Replace `src/nanoscope/tokenizer/__init__.py` with:

```python
"""Byte-level BPE tokenizer."""

from nanoscope.tokenizer.tokenizer import Tokenizer
from nanoscope.tokenizer.train import train

__all__ = ["Tokenizer", "train"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: 12 passed.

- [ ] **Step 6: Run the full check**

Run: `make check`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/nanoscope/tokenizer/tokenizer.py src/nanoscope/tokenizer/__init__.py tests/test_tokenizer.py
git commit -m "feat: add byte-level encoding and decoding

decode(encode(b)) == b holds over st.binary() with no exclusions, which is
only possible because encode never emits the special token. The str layer
is a surrogatepass wrapper, so lone surrogates round-trip instead of
raising -- the case a strict-UTF-8 design would have to exclude from its
own test strategy."
```

---

## Task 5: Serialization

**Files:**
- Modify: `src/nanoscope/tokenizer/vocab.py` (add the `TokenizerFile` model)
- Modify: `src/nanoscope/tokenizer/tokenizer.py` (add `save` and `load`)
- Test: `tests/test_tokenizer.py` (append)

**Interfaces:**
- Consumes: `Tokenizer` from Task 4; `PATTERN_SOURCE` from Task 1.
- Produces: `TokenizerFile` (Pydantic model) in `vocab.py`; `Tokenizer.save(path: Path) -> None` and `Tokenizer.load(path: Path) -> Tokenizer` (classmethod). Task 6 calls `save`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tokenizer.py`:

```python
def test_a_saved_tokenizer_loads_back_identically(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    TRAINED.save(path)
    loaded = Tokenizer.load(path)
    data = b"the cat sat on the mat \xff\x00"
    assert loaded.encode(data) == TRAINED.encode(data)
    assert loaded.vocab_size == TRAINED.vocab_size


def test_the_corpus_hash_survives_a_round_trip(tmp_path) -> None:
    """Published compression numbers have to trace to the exact training bytes."""
    path = tmp_path / "tokenizer.json"
    Tokenizer([], corpus_sha256="deadbeef").save(path)
    assert Tokenizer.load(path).corpus_sha256 == "deadbeef"


def test_load_rejects_a_merge_that_references_an_undefined_id(tmp_path) -> None:
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


def test_load_rejects_a_file_written_with_a_different_split_pattern(tmp_path) -> None:
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
```

Add these imports to the top of `tests/test_tokenizer.py`:

```python
import json

import pytest
from pydantic import ValidationError

from nanoscope.tokenizer.pretokenize import PATTERN_SOURCE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: failures with `AttributeError: 'Tokenizer' object has no attribute 'save'`

- [ ] **Step 3: Add the serialization model**

Append to `src/nanoscope/tokenizer/vocab.py`:

```python
from pydantic import BaseModel, field_validator


class TokenizerFile(BaseModel):
    """On-disk representation of a trained tokenizer.

    JSON rather than a binary format so a merge table is diffable in review and
    a corrupted one is readable by eye.
    """

    version: int = 1
    pattern: str
    special_tokens: dict[str, int]
    corpus_sha256: str | None = None
    merges: list[tuple[int, int]]

    @field_validator("merges")
    @classmethod
    def _merges_only_reference_defined_ids(
        cls, merges: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """A merge at rank k may reference only ids below FIRST_MERGE_ID + k.

        Without this, a truncated or hand-edited file loads into a tokenizer
        whose vocabulary table is subtly wrong -- which shows up as degraded
        compression, not as an error.
        """
        for rank, (first, second) in enumerate(merges):
            limit = FIRST_MERGE_ID + rank
            for token_id in (first, second):
                if not 0 <= token_id < limit:
                    raise ValueError(
                        f"merge at rank {rank} references id {token_id}, "
                        f"which is outside the defined range [0, {limit})"
                    )
        return merges
```

The `from pydantic import ...` line belongs at the top of the file with the other imports, not inline. ruff's isort rules will flag it otherwise.

- [ ] **Step 4: Add `save` and `load`**

Append these methods to the `Tokenizer` class in `src/nanoscope/tokenizer/tokenizer.py`:

```python
    def save(self, path: Path) -> None:
        document = TokenizerFile(
            pattern=PATTERN_SOURCE,
            special_tokens={END_OF_TEXT: END_OF_TEXT_ID},
            corpus_sha256=self.corpus_sha256,
            merges=self._merges,
        )
        path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        document = TokenizerFile.model_validate_json(path.read_text(encoding="utf-8"))
        if document.pattern != PATTERN_SOURCE:
            raise ValueError(
                "tokenizer file was trained with a different split pattern; "
                "its merge table is not valid under this one"
            )
        return cls(merges=list(document.merges), corpus_sha256=document.corpus_sha256)
```

Extend the imports at the top of `tokenizer.py` to:

```python
from pathlib import Path

from nanoscope.tokenizer.pretokenize import PATTERN_SOURCE, pretokenize
from nanoscope.tokenizer.train import Pair
from nanoscope.tokenizer.vocab import (
    BYTE_TOKENS,
    END_OF_TEXT,
    END_OF_TEXT_ID,
    FIRST_MERGE_ID,
    TokenizerFile,
)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer.py -q`
Expected: 16 passed.

- [ ] **Step 6: Run the full check**

Run: `make check`
Expected: all clean.

- [ ] **Step 7: Commit**

```bash
git add src/nanoscope/tokenizer/vocab.py src/nanoscope/tokenizer/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: persist tokenizers as validated JSON

Load enforces that a merge at rank k references only ids below 257+k, and
that the file's split pattern matches this build's. Both failures would
otherwise produce a tokenizer that works but compresses worse, which is
the failure mode no round-trip test can see.

The corpus sha256 travels with the file so any published compression
number traces to the exact bytes it was trained on."
```

---

## Task 6: CLI verb

**Files:**
- Modify: `src/nanoscope/cli.py`
- Test: `tests/test_tokenizer_cli.py`

**Interfaces:**
- Consumes: `Tokenizer`, `train` from `nanoscope.tokenizer`; `DEFAULT_VOCAB_SIZE`, `FIRST_MERGE_ID` from `nanoscope.tokenizer.vocab`.
- Produces: the `nanoscope tokenizer train --input PATH --output PATH [--vocab-size N]` command.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tokenizer_cli.py`:

```python
"""End-to-end test of the `tokenizer train` verb."""

from typer.testing import CliRunner

from nanoscope.cli import app
from nanoscope.tokenizer import Tokenizer
from nanoscope.tokenizer.vocab import FIRST_MERGE_ID

runner = CliRunner()


def test_train_writes_a_loadable_tokenizer(tmp_path) -> None:
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


def test_train_records_the_corpus_hash(tmp_path) -> None:
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tokenizer_cli.py -q`
Expected: failures, exit code 2 with "No such command 'tokenizer'"

- [ ] **Step 3: Add the CLI verb**

In `src/nanoscope/cli.py`, extend the imports to:

```python
import hashlib
from pathlib import Path
from typing import Annotated

import typer

from nanoscope import __version__
from nanoscope.tokenizer import Tokenizer, train
from nanoscope.tokenizer.vocab import DEFAULT_VOCAB_SIZE, FIRST_MERGE_ID
```

Then append, after the existing `version` command:

```python
tokenizer_app = typer.Typer(
    help="Train the byte-level BPE tokenizer.",
    no_args_is_help=True,
)
app.add_typer(tokenizer_app, name="tokenizer")


@tokenizer_app.command("train")
def tokenizer_train(
    input_path: Annotated[Path, typer.Option("--input", help="Training corpus.")],
    output_path: Annotated[Path, typer.Option("--output", help="Destination JSON file.")],
    vocab_size: Annotated[
        int, typer.Option("--vocab-size", help="Total vocabulary size.")
    ] = DEFAULT_VOCAB_SIZE,
) -> None:
    """Train a tokenizer on a corpus file and write it to disk.

    The corpus is read whole rather than streamed: chunking a stream at
    arbitrary boundaries splits pre-tokens across them and perturbs the pair
    counts. See section 10 of the tokenizer spec.
    """
    corpus = input_path.read_bytes()
    digest = hashlib.sha256(corpus).hexdigest()
    merges = train(corpus, vocab_size)
    Tokenizer(merges, corpus_sha256=digest).save(output_path)
    typer.echo(
        f"wrote {output_path}: {len(merges)} merges, "
        f"vocab {FIRST_MERGE_ID + len(merges)}, corpus sha256 {digest}"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tokenizer_cli.py -q`
Expected: 3 passed.

- [ ] **Step 5: Run the full check**

Run: `make check`
Expected: all clean, including the pre-existing `tests/test_cli.py`.

- [ ] **Step 6: Commit**

```bash
git add src/nanoscope/cli.py tests/test_tokenizer_cli.py
git commit -m "feat: add the tokenizer train CLI verb

Reads the corpus whole and records its sha256 in the artifact, so a
published compression number traces to exact bytes. Streaming was rejected:
chunking at arbitrary boundaries splits pre-tokens and perturbs pair counts."
```

---

## Task 7: Decision record and handoff

**Files:**
- Modify: `docs/DECISIONS.md`
- Modify: `NEXT.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Append ADR-0004 to `docs/DECISIONS.md`**

```markdown
---

## ADR-0004: Bytes as the tokenizer's domain, and two trainers

**Date:** 2026-08-11
**Status:** accepted

**Context.** Two independent questions had to be settled before writing a
byte-level BPE tokenizer. First, what type the round-trip law quantifies over:
Python `str` can hold lone surrogates that strict UTF-8 refuses to encode, so a
`str`-domain tokenizer either raises on them or excludes them from its test
strategy. Second, how to know the trainer is correct: a BPE trainer that
miscounts pairs produces a *worse* merge table, not an invalid one, so the
tokenizer still round-trips perfectly while the compression ratio this repo
publishes is quietly wrong.

**Decision.** `bytes` is the fundamental domain. `decode(encode(b)) == b` is
tested over `st.binary()` with no exclusions, and `str` support is a thin
`surrogatepass` wrapper that is total on `str`, so lone surrogates round-trip.
For the trainer, ship two implementations: a naive one that rescans every pair
on every merge, and an indexed one that tracks which words contain each pair.
A Hypothesis differential test asserts they produce identical merge tables. The
slow implementation is correct by inspection and certifies the fast one.

Pre-tokenization uses `tiktoken`'s cl100k_base split pattern verbatim, which
adds `regex` as a runtime dependency for `\p{L}` and `\p{N}` support.

**Alternatives considered.**
- *A `str`-only API with strict UTF-8.* Mirrors `tiktoken`'s public surface and
  is the simplest. Rejected because the Hypothesis strategy would need
  surrogates excluded, and DESIGN.md section 6 names lone surrogates as exactly
  where naive implementations break. A test that excludes its hardest input is
  not evidence.
- *The indexed trainer alone.* Less code. Rejected for the reason above: no
  round-trip property can detect a stale pair count, so the alternative to a
  differential test is no test at all for the thing that determines the
  headline compression number.
- *The naive trainer alone.* Rejected on retraining cost, which makes any later
  ablation over vocabulary size expensive.
- *An approximation of the split pattern using the standard library `re`.*
  Avoids the `regex` dependency, at the cost of `[^\W\d_]` standing in for
  `\p{L}` -- close, but it also admits Nl and No characters. Rejected because
  DESIGN.md section 6 promises a compression comparison against `tiktoken`, and
  a different splitter would make that table report the combined effect of two
  splitters and two merge tables rather than isolating the merge table.

**Trade-off.** Writing two trainers is roughly 190 lines where one would be 150,
and the differential test only proves they agree, not that either is optimal.
That is the right guarantee to buy: BPE has no contested definition of optimal,
but it does have a well-known failure mode in incremental bookkeeping, and this
closes it.
```

- [ ] **Step 2: Rewrite `NEXT.md`**

The current file lists a GitHub push blocker that is stale — the `workflow`
scope went through and CI is green. Replace the whole file with:

```markdown
# Next unit of work

Tokenizer landed 2026-08-11: pre-tokenizer, both trainers, encoder, decoder,
serialization, and the `tokenizer train` CLI verb. `make check` green.

## Next, in order

1. **Data pipeline.** TinyStories download, tokenize, memmap shards. Held-out
   validation split fixed by seed so every ablation sees identical data in
   identical order.
2. **Model.** Components in dependency order: RMSNorm, RoPE, attention
   (GQA + SDPA), SwiGLU MLP, block, transformer. Every component behind a
   config flag from the start, since retrofitting toggles later is how ablation
   harnesses rot.
3. **Training loop**, then the **calibration run** (200 steps) that sets every
   downstream token budget.

## Carried forward from the tokenizer

- **Train the real artifact.** The tokenizer is corpus-agnostic and no trained
  tokenizer is committed yet. It gets trained once the TinyStories split exists.
- **The `tiktoken` comparison from DESIGN.md section 6** — compression ratio in
  bytes per token, plus encode throughput. Needs the held-out validation split
  to be meaningful. Add `tiktoken` to the dev group then.
- **Corpus memory.** `train` takes the whole corpus as `bytes` and holds the
  decoded `str` alongside it during pre-tokenization. If the TinyStories split
  is too large for 16 GB, train on a seeded, documented subsample and record
  its sha256. Do not switch to a streaming trainer: chunking at arbitrary
  boundaries splits pre-tokens and perturbs the pair counts. Decided by
  measurement, not now.

## Open decision, deferred until calibration produces real numbers

Whether to spend ~25 minutes on one repeat of the baseline at a different seed,
to establish a noise floor for the whole ablation table. Multi-seed replication
was already declined as too expensive; this is the cheaper version and applies
to every row rather than two. Revisit once a run's true cost is measured.

## Standing constraints

- No `Co-Authored-By` trailers. Commits are authored by Shivank Pandey alone.
- Never run git write commands from `/Users/shivpanndey05` or any uninitialised
  subdirectory: the home directory is itself a repo holding ~40 unpushed commits
  of unrelated work.
- No fabricated numbers. Calibration measures throughput; it does not get
  estimated.
```

- [ ] **Step 3: Verify the full suite one more time**

Run: `make check`
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add docs/DECISIONS.md NEXT.md
git commit -m "docs: record ADR-0004 and hand off to the data pipeline

Also drops the stale GitHub push blocker from NEXT.md: the workflow scope
went through and CI is green.

Carries three items forward explicitly rather than dropping them: the
trained artifact, the tiktoken comparison from DESIGN.md 6, and the corpus
memory question, all of which need the data pipeline first."
```

- [ ] **Step 5: Push**

```bash
git push
```

Then confirm CI is green: `gh run list --limit 1`

---

## Verification Summary

Every spec requirement maps to a task:

| Spec section | Task |
|---|---|
| §1 scope and deferrals | 7 (NEXT.md carries them forward) |
| §2 vocabulary layout | 2 (`vocab.py`), pinned by tests in 4 |
| §3 module layout | 1-5 (plus the `vocab.py` deviation, recorded above) |
| §4 public interface | 4 (encode/decode/str wrappers), 5 (save/load) |
| §5.1 pre-tokenization | 1 |
| §5.2 training, tie-break, two implementations | 2, 3 |
| §5.3 encoding and the chunk cache | 4 |
| §5.4 decoding | 4 |
| §6 serialization and the integrity rule | 5 |
| §7 properties 1, 2, 6 | 4 |
| §7 property 3 | 1 |
| §7 property 4 | 3 |
| §7 property 5 and idempotence | 1 (idempotence), 4 (chunk-locality) |
| §7 property 7 | 5 |
| §7 unit tests | 1, 2, 4, 5 |
| §8 CLI | 6 |
| §9 dependency changes | 1 (`regex`), 7 (`tiktoken` deferral recorded) |
| §10 corpus memory limitation | 6 (docstring), 7 (NEXT.md) |
| §11 success criteria | verified by `make check` in every task |
