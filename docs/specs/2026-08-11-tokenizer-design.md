# Spec: byte-level BPE tokenizer

Status: approved
Author: Shivank Pandey
Date: 2026-08-11
Implements: DESIGN.md §6

## 1. Scope

This unit ships a corpus-agnostic byte-level BPE tokenizer: library, CLI verb, and
test suite. It takes any file or byte string as training input.

DESIGN.md orders the tokenizer before the data pipeline, but training a tokenizer
needs a corpus and the corpus arrives with the pipeline. Rather than reorder, the
tokenizer is built to depend on no particular corpus. Two things therefore defer,
and are recorded here so neither looks quietly dropped:

- **The trained TinyStories artifact.** Produced once the data pipeline lands.
- **The `tiktoken` comparison promised in DESIGN.md §6** — compression ratio in
  bytes per token, plus encode throughput. It is only meaningful on the held-out
  validation split, which does not exist yet. `tiktoken` joins the dev dependency
  group at that point, not now.

Trainer wall-clock is measured when the real training run happens. It is not
estimated here, per the project's no-fabrication rule.

## 2. Vocabulary layout

Vocabulary size 8192, laid out so that a byte token's id equals its byte value.
That invariant makes encoder bugs visible by inspection.

```
id     0..255   raw bytes, id == byte value
id       256    <|endoftext|>
id  257..8191   7,935 learned merges, in rank order
                ------------------------------------
                8,192 total
```

One special token, not four. Byte-level encoding makes `UNK` unreachable by
construction; packed fixed-length training sequences make `PAD` unused; and a
single separator makes a distinct `BOS` redundant for a document-stream corpus.
`<|endoftext|>` serves as both the TinyStories document separator and the
generation stop signal that Project 4 (`inferno`) will need.

`encode` never emits id 256. The data pipeline appends it directly. This is what
keeps the round-trip law in §6 total: no byte string can produce a special token,
so no byte string can fail to survive the round trip. `decode` renders id 256 as
its literal spelling, which means `encode ∘ decode` is not the identity on
sequences containing specials. That is expected, and documented rather than
tested away.

## 3. Module layout

```
src/nanoscope/tokenizer/
  __init__.py       public exports: Tokenizer, train
  pretokenize.py    split pattern and pretokenize(bytes) -> list[bytes]
  train.py          train_naive (oracle), train_indexed (shipped)
  tokenizer.py      Tokenizer: encode/decode, encode_str/decode_str, save/load
```

`train.py` does not import `tokenizer.py`. Training produces a merge list;
`Tokenizer` consumes one. Keeping that direction one-way means the trainer's
differential test in §6 exercises the trainer alone, with no encoder in the path
to muddy a failure.

## 4. Public interface

```python
class Tokenizer:
    def encode(self, data: bytes) -> list[int]: ...
    def decode(self, ids: list[int]) -> bytes: ...
    def encode_str(self, text: str) -> list[int]: ...
    def decode_str(self, ids: list[int]) -> str: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Tokenizer": ...

def train(corpus: bytes, vocab_size: int) -> list[tuple[int, int]]: ...
```

`train` is the public name and is an alias for `train_indexed` (§5.2).
`train_naive` is exported from `train.py` but not from the package, because its
only caller is the differential test.

`bytes` is the fundamental domain, because a byte-level tokenizer's real invariant
is over bytes and nothing else. The `str` methods are thin wrappers:

```python
encode_str(s) == encode(s.encode("utf-8", errors="surrogatepass"))
decode_str(t) == decode(t).decode("utf-8", errors="surrogatepass")
```

`surrogatepass` is total on `str`, so lone surrogates round-trip rather than
raising. The alternative — a strict-UTF-8 `str` API with surrogates excluded from
the test strategy — is exactly the happy-path carve-out DESIGN.md §6 names as
where naive implementations break.

## 5. Algorithms

### 5.1 Pre-tokenization

Merges never cross a pre-token boundary. The split pattern is cl100k_base's,
taken verbatim from `tiktoken` (MIT) and attributed in the source:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Using `tiktoken`'s pattern rather than an approximation is a deliberate choice
about what §6's compression comparison measures. With a different splitter, that
table would report the combined effect of two different splitting rules and two
different merge tables. With an identical splitter it isolates the merge table,
which is the part this repo actually built.

The pattern needs `\p{L}` and `\p{N}`, which the standard library `re` does not
support, so `regex` becomes a runtime dependency. Matching uses `finditer` and
takes `.group()`, not `findall`, so the inline group cannot alter what is
returned.

The pattern operates on `str` while the domain is `bytes`. Bridged by
`surrogateescape`, which is total over all byte strings and byte-exact on
re-encode:

```python
text = data.decode("utf-8", errors="surrogateescape")
chunks = [m.group().encode("utf-8", errors="surrogateescape")
          for m in PATTERN.finditer(text)]
```

The pattern's branches cover every character, so the chunks partition the input.
Property 3 in §6 asserts this rather than assuming it.

### 5.2 Training

Count pre-token chunk frequencies into a `dict[tuple[int, ...], int]`, then
repeatedly merge the most frequent adjacent pair. Ties are broken explicitly:

```python
best = min((-count, pair) for pair, count in counts.items())[1]
```

Highest count wins; equal counts go to the lexicographically smallest pair.
Without a fixed tie-break the two trainers below can legitimately diverge, and
the differential test that justifies the whole approach becomes meaningless.

Two implementations:

- `train_naive` recounts every pair from scratch on each merge. Short and
  obviously correct. It is the test oracle, not the shipped path.
- `train_indexed` additionally maintains `pair -> set[word_id]`, so each merge
  rescans only the words that contained the merged pair.

The reason for writing both: incremental count maintenance fails in a way that no
round-trip test can catch. A stale pair count yields a *worse* merge table, not
an invalid one, so the tokenizer still round-trips perfectly while the compression
number this repo publishes is silently wrong. A differential test against a slow
implementation that is correct by inspection is the cheap way to rule that out.

### 5.3 Encoding

Per chunk, repeatedly apply the lowest-rank merge present until none applies.
Chunks are short by construction, so the quadratic inner loop is bounded.

An instance-level `dict[bytes, list[int]]` caches chunk results. TinyStories has
few unique chunks, so this is most of the corpus-pass throughput. The cache clears
at 100,000 entries: arbitrary binary input has unbounded chunk diversity and would
otherwise grow it without limit.

### 5.4 Decoding

Build `vocab: list[bytes]` once at construction — `bytes([i])` for ids 0-255, the
literal spelling for 256, and `vocab[a] + vocab[b]` for each merge in rank order.
Decoding is then a lookup and a join.

## 6. Serialization

JSON, human-diffable, validated on load with Pydantic, which is already a
dependency.

```json
{
  "version": 1,
  "pattern": "...",
  "special_tokens": {"<|endoftext|>": 256},
  "corpus_sha256": "...",
  "merges": [[104, 101], [257, 108]]
}
```

`load` enforces one integrity rule: a merge at rank *k* may reference only ids
`< 257 + k`. A truncated or hand-edited file then fails loudly instead of
constructing a tokenizer whose vocabulary table is subtly wrong.

`corpus_sha256` records the training corpus, so any compression number published
in the README traces to the exact bytes that produced it.

## 7. Test plan

Property tests, with Hypothesis:

| # | Property | Strategy |
|---|---|---|
| 1 | `decode(encode(b)) == b` | `st.binary()`, total, no carve-outs |
| 2 | `decode_str(encode_str(s)) == s` | `st.text()` including lone surrogates |
| 3 | `b"".join(pretokenize(b)) == b` | `st.binary()`, the split loses nothing |
| 4 | `train_indexed(c, k) == train_naive(c, k)` | random small corpora × merge counts |
| 5 | `encode(b) == concat(encode(c) for c in pretokenize(b))` | merges never cross a chunk boundary |
| 6 | every emitted id lies in `[0, vocab_size)` | `st.binary()` |
| 7 | `load(save(t))` encodes identically to `t` | round-trip through disk |

Property 5 carries a second assertion worth stating: since `encode` already
pre-tokenizes internally, the property holds only if pre-tokenization is
idempotent — that is, if re-splitting a chunk yields that chunk unchanged. The
cl100k pattern is expected to satisfy this, but it is not self-evident from
reading it. A failure here is a real finding about the pattern, not a bug in the
encoder, and the test should be read that way.

Unit tests:

- A zero-merge tokenizer satisfies `encode(b) == list(b)`, pinning the id-equals-
  byte-value invariant from §2.
- A hand-computed merge table on a tiny fixture, checked against `train` output.
- `load` rejects a merge that forward-references an id not yet defined.
- Emoji, combining characters, and a lone surrogate as explicitly named regression
  cases, alongside the property tests that already cover them. Named cases make a
  regression legible in the test report rather than surfacing as a Hypothesis
  counterexample.

## 8. CLI

One new verb, following the standing rule in `cli.py` that a verb appears only
once the thing behind it works:

```
nanoscope tokenizer train --input PATH --vocab-size 8192 --output PATH
```

No `encode` or `inspect` verb. Nothing needs them yet.

## 9. Dependency changes

- `regex` added to runtime dependencies, for `\p{L}` and `\p{N}` support (§5.1).
- `tiktoken` deferred to the dev group, when the comparison in §1 becomes
  measurable.

## 10. Known limitation: the corpus is held in memory

`train` takes `bytes`, not a stream. Counting chunk frequencies is streaming-
friendly, so an iterator interface would be little extra code, but chunking a
stream at arbitrary boundaries splits pre-tokens across those boundaries and
perturbs the counts. Taking the whole corpus is exact.

The cost is memory: the corpus is held as `bytes` and again as the
`surrogateescape`-decoded `str` during pre-tokenization, on a 16 GB machine that
is also the training machine. If the TinyStories split turns out to be too large
for that, the resolution is to train on a **seeded, documented subsample** and
record its `corpus_sha256`, not to switch to streaming and accept boundary
artifacts. Subsampling for BPE training is standard practice; silently miscounting
pairs is not.

Which of the two applies is decided by measurement when the data pipeline lands,
not now.

## 11. Success criteria

1. All seven properties in §7 hold, with `st.binary()` and `st.text()` unrestricted.
2. `train_indexed` and `train_naive` agree on every generated case.
3. `make check` green: ruff, ruff format, mypy strict, pytest.
4. A tokenizer survives save and load with identical behaviour.
5. `nanoscope tokenizer train` produces a valid artifact from a plain text file.
