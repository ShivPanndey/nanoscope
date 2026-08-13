# Data pipeline implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a text corpus into memory-mapped `uint16` token shards with a
seed-reproducible held-out split, implementing
`docs/specs/2026-08-13-data-pipeline-design.md`.

**Architecture:** Four stages behind an `Iterator[bytes]` seam. Everything downstream of
that seam is testable with synthetic documents and no download, which is where nearly
all of the test suite lives. Only the fetch stage needs the network.

**Tech Stack:** Python 3.13, numpy (memmap), Pydantic v2 (manifest), Hypothesis
(property tests), Typer (CLI), uv, ruff, mypy strict, pytest.

## Global Constraints

- Python `>=3.13`. Run everything through `uv run`.
- `make check` must be green before any task is considered done: ruff,
  `ruff format --check`, mypy strict, pytest.
- **`ruff format` also checks fenced Python blocks inside markdown.** This plan
  contains them. Run `make check` after any edit to this file.
- ruff line-length is 100. The whole `RUF` ruleset is selected, `ANN` is on, so every
  function needs full annotations including `-> None` on tests. `tests/**` waives
  `ANN001` for ruff only; **mypy strict does not honour that waiver**, so a `tmp_path`
  parameter still needs `: Path`.
- **Do not write `zip(x, x[1:])`.** RUF007 rejects it in favour of `itertools.pairwise`.
  The tokenizer plan mandated that form and it failed to lint in three separate tasks.
  Any literal code in this plan should be run through `make check` before it is trusted.
- mypy is `strict` with `warn_unreachable` and `disallow_any_generics`. No bare
  `dict`/`list` annotations.
- **Commits are authored by Shivank Pandey alone. Never add a `Co-Authored-By` trailer**
  or any AI attribution.
- Never run git write commands from `/Users/shivpanndey05` or any uninitialised
  subdirectory. All git commands run from `/Users/shivpanndey05/dev/portfolio/nanoscope`.
- **No fabricated numbers.** Every number in code, docs, help text, or a commit message
  must be one that was observed. The chunk-length ceiling in Task 1 and the throughput
  figure in Task 7 are measurements, not estimates.
- **CLI output assertions must be checked with `FORCE_COLOR=1`.** Rich colourises in CI
  and not in a plain local run, and a helper that strips non-alphanumerics turns an ANSI
  code into a fragment like `36m` that splits words. This broke CI once already.
- Token ids are `uint16` on disk. Vocabulary is 8192, so they fit with room to spare.
- Tests live flat in `tests/`. Do not create `tests/` subpackages.
- **Ask before any download of the real corpus, or any run expected to exceed 15
  minutes.** Tasks 1 through 6 need neither. Task 7 needs both and is gated.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/nanoscope/data/__init__.py` | Public exports: `ShardedTokens`, `prepare` |
| `src/nanoscope/data/manifest.py` | The `Manifest` Pydantic model and its digest helpers |
| `src/nanoscope/data/documents.py` | `iter_documents(path) -> Iterator[bytes]` |
| `src/nanoscope/data/shards.py` | `ShardWriter`, `ShardedTokens` |
| `src/nanoscope/data/prepare.py` | `prepare(...)`, the stage that wires the others together |
| `src/nanoscope/data/fetch.py` | Corpus download and digest verification |
| `src/nanoscope/cli.py` | *modify*: add the `data prep` sub-app |
| `tests/test_documents.py` | Document iteration, including boundary cases |
| `tests/test_shards.py` | Round-trip, straddling windows, shard boundaries |
| `tests/test_manifest.py` | Digest mismatch rejection, round-trip |
| `tests/test_prepare.py` | Split partition properties, end to end on synthetic docs |
| `tests/test_data_cli.py` | The `data prep` verb |
| `docs/DECISIONS.md` | *modify*: ADR-0005 |
| `NEXT.md` | *modify*: hand off to the model |

---

## Task 1: Measure the pre-token chunk length distribution

This task writes no shipped code. It produces the measurement that Task 4's ceiling
depends on, so that the ceiling is a number somebody observed rather than a number
somebody liked.

**Why first:** the design spec deliberately refuses to guess `max_chunk_bytes`. Section
5 records that `encode` is O(L squared) in chunk length, with 128 KB taking 364 seconds,
and that natural prose never approaches it. That claim needs a number attached.

- [ ] **Step 1: Write a throwaway measurement script**

Put it outside the repo, in a scratch directory. It is not committed.

For every `.md`, `.py`, and `.toml` file in the repo, run `pretokenize` over the bytes
and accumulate a histogram of chunk lengths. Report the p50, p90, p99, p99.9, and
maximum chunk length, the total bytes scanned, and the single longest chunk with the
file it came from.

The repo's own prose and code stand in for the corpus, which is not downloaded yet.
That substitution must be stated wherever the number is used.

- [ ] **Step 2: Record the measurement**

Append the observed distribution to `docs/specs/2026-08-13-data-pipeline-design.md`
section 5, labelled with what was actually measured, in this shape:

> Measured over the repository's own markdown, Python, and TOML files (N files, N
> bytes, N chunks): p50 N, p90 N, p99 N, p99.9 N, longest N bytes. TinyStories prose is
> expected to be shorter-tailed than source code, which contains long identifiers and
> URLs, so this is a conservative stand-in until the corpus is available.

Use the numbers observed. Do not round them into something tidier.

- [ ] **Step 3: Choose the ceiling and justify it**

Pick `DEFAULT_MAX_CHUNK_BYTES` as a round number comfortably above the observed maximum
and comfortably below where the quadratic cost becomes noticeable. The design spec's
measurements bound it from above: 2 KB costs 0.085s per chunk, which is already too slow
to pay often.

State the reasoning in the same section. A reader must be able to see both bounds.

- [ ] **Step 4: Commit**

Docs only. `make check` must pass, since the spec is markdown and may contain fences.

---

## Task 2: The manifest model

**Files:** create `src/nanoscope/data/__init__.py`, `src/nanoscope/data/manifest.py`;
test `tests/test_manifest.py`.

**Interfaces:**
- Consumes: nothing.
- Produces: `Manifest` and `ShardEntry` from `nanoscope.data.manifest`, plus
  `sha256_file(path: Path) -> str`.

Follow `src/nanoscope/tokenizer/vocab.py`'s `TokenizerFile` as the precedent. That model
learned two things the hard way, and both apply here:

- `model_config = ConfigDict(extra="forbid")`, so a misspelled field is rejected rather
  than silently dropped to its default.
- A `version` field is worthless unless something checks it. Whatever reads this
  manifest must reject a version it does not understand, naming both versions.

Fields, per design spec section 7: `version`, `tokenizer_sha256`, `source_sha256`,
`seed`, `val_fraction`, `max_chunk_bytes_observed`, `nanoscope_version`, and a list of
`ShardEntry` carrying `name`, `split`, `tokens`, and `sha256`.

Tests: round-trip through JSON; an unknown field is rejected; a wrong `version` is
rejected; `sha256_file` matches a digest computed another way.

---

## Task 3: Document iteration

**Files:** create `src/nanoscope/data/documents.py`; test `tests/test_documents.py`.

**Interfaces:**
- Produces: `iter_documents(path: Path) -> Iterator[bytes]`.

Yields documents one at a time without holding the file in memory. The separator and
the source format are properties of the corpus, so keep the reader dumb and explicit:
one document per line, yielded as `bytes` with the trailing newline stripped.

Design spec section 4 is the constraint that matters. Document boundaries are safe
encode boundaries because `\s*[\r\n]+` is one of the split pattern's alternatives, so a
newline is always a chunk boundary. Nothing here may split at an arbitrary byte offset.

Tests: an empty file yields nothing; a file without a trailing newline still yields its
last document; a blank line yields an empty document rather than being skipped; the
concatenation of the yielded documents plus separators reproduces the file exactly.
Include a property test over synthetic files built from `st.lists(st.binary())` joined
by newlines, asserting that round trip.

---

## Task 4: Shard writing and reading

**Files:** create `src/nanoscope/data/shards.py`; test `tests/test_shards.py`.

**Interfaces:**
- Consumes: `ShardEntry` from Task 2.
- Produces: `ShardWriter` and `ShardedTokens`.

`ShardWriter` appends token ids to fixed-size shards, rolling over to a new file at
`shard_tokens`, and returns the `ShardEntry` list it wrote. `ShardedTokens` opens a
split's shards as `numpy.memmap` and serves `window(start, length) -> npt.NDArray`
across shard seams, so a caller never learns where the boundaries are.

Headerless little-endian `uint16`, per design spec section 7. Assert on write that
every id fits, and raise naming the offending id if not: an id above 65535 would wrap
silently and poison the corpus in a way no later stage could detect.

The straddling window is the case to get right, and it is the case a naive
implementation gets wrong. Test it directly: build a known ramp of ids across at least
three shards, then assert that every window straddling each seam equals the same window
taken from an unsharded array. Property-test it over random start offsets and lengths
rather than only at the seams.

---

## Task 5: Preparation and the split

**Files:** create `src/nanoscope/data/prepare.py`; modify
`src/nanoscope/data/__init__.py`; test `tests/test_prepare.py`.

**Interfaces:**
- Consumes: everything from Tasks 2 through 4, plus `Tokenizer` and `END_OF_TEXT_ID`.
- Produces: `prepare(...) -> Manifest`, exported from `nanoscope.data`.

The split is by document and deterministic given a seed, per design spec section 6:
enumerate documents, permute the indices with `numpy.random.default_rng(seed)`, take the
first `val_fraction` of the permuted order as validation.

Each document is encoded and terminated with `END_OF_TEXT_ID`. Track the longest
pre-token chunk seen; if any exceeds `max_chunk_bytes`, raise naming the document index
and the observed length, per design spec section 5.

Tests, all on synthetic documents with a small hand-built tokenizer, no download:
- The split is a partition. Every document index appears in exactly one side, and the
  union is the whole set. Property-test over document counts and fractions.
- The same seed reproduces the same assignment; a different seed generally does not.
- Decoding a split's shards reproduces its documents, with `END_OF_TEXT_ID` at each
  boundary.
- A document containing an over-long chunk is rejected, and the error names the index.
- `val_fraction` of 0 and of 1 behave sensibly rather than producing an empty memmap
  that fails later.

---

## Task 6: The `data prep` CLI verb

**Files:** modify `src/nanoscope/cli.py`; test `tests/test_data_cli.py`.

**Interfaces:**
- Produces: `nanoscope data prep --source PATH --tokenizer PATH --output DIR [--val-fraction F] [--seed N] [--shard-tokens N] [--max-chunk-bytes N] [--limit N]`.

Follow the `tokenizer train` verb exactly, including the lessons its review round
forced:

- Validate **every** path before doing any work. `--source` and `--tokenizer` take
  `exists=True, dir_okay=False, readable=True`. `--output` is a directory. A full run is
  expensive and a typo must not cost it.
- Bind the object and print its own accessors rather than recomputing a number the
  library already owns.
- Bound numeric options with `min=` so a typo becomes an argument error rather than a
  silently useless artifact.

`--limit` caps documents processed, which is what makes an end-to-end CLI test possible
without a corpus.

Tests: a successful small run writes a loadable shard set whose manifest round-trips;
each bad-path case exits 2 with a named error and writes nothing. **Assert CLI output
with a helper that strips ANSI sequences as whole sequences before filtering, and run
the file once with `FORCE_COLOR=1`** before believing it.

---

## Task 7: The real corpus

**GATED.** This task needs a large download and a run likely to exceed 15 minutes. Stop
and ask before starting it. Tasks 1 through 6 must be complete and merged first.

**Files:** create `src/nanoscope/data/fetch.py`; modify `docs/DECISIONS.md`, `NEXT.md`.

- [ ] Fetch TinyStories, record its sha256, verify on every later use.
- [ ] Train the tokenizer on the training split. This is the artifact the tokenizer
      work has been waiting on. Record its wall clock, measured.
- [ ] Measure the real chunk length distribution and compare it against Task 1's
      stand-in. If the stand-in was optimistic, say so and revise the ceiling.
- [ ] Run `data prep` on a `--limit`ed sample first, measure throughput, and extrapolate
      the full run's cost **before** committing to it.
- [ ] Run the full preparation. Record token counts.
- [ ] The `tiktoken` comparison owed from DESIGN.md section 6: compression ratio in
      bytes per token on the held-out split, plus encode throughput. Add `tiktoken` to
      the dev group. Report both numbers even where they are unflattering.
- [ ] ADR-0005 recording the streaming, splitting, and ceiling decisions.
- [ ] Rewrite `NEXT.md` to hand off to the model.

---

## Verification Summary

Every task ends with `make check` green. Tasks 1 through 6 require no network and no
long compute, so the whole pipeline is verifiable offline against synthetic documents
before any corpus exists. That is deliberate: it is what lets the expensive, gated task
be a measurement exercise rather than a debugging one.
