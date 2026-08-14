# Data pipeline design

**Status:** proposed, awaiting review
**Date:** 2026-08-13
**Implements:** DESIGN.md section 7 (`data/`), and the `data prep` CLI verb

## 1. Goal

Turn the TinyStories corpus into memory-mapped token shards that the training loop
can read at speed, plus a held-out validation split that is fixed by seed so every
ablation sees identical data in identical order.

The pipeline is the first consumer of the tokenizer, and the first code in this
project to touch a real corpus rather than a test fixture.

### Non-goals

- No streaming or distributed anything. One machine, one process.
- No on-the-fly tokenization during training. Tokenizing once and memory-mapping the
  result is the whole point: it moves a fixed cost out of the training loop so that
  throughput measurements describe the model rather than the data path.
- No dataset augmentation, filtering, or dedup. The corpus is taken as published, so
  the ablation study measures architecture rather than data cleaning.

## 2. What the training loop actually needs

The loop needs to draw a batch of `(batch, seq_len)` int64 tensors uniformly at random
from the training tokens, cheaply, thousands of times. That is one requirement, and it
drives every decision here:

- A flat array of token ids, not a list of documents. Sampling a window that straddles
  a document boundary is fine and is standard practice; it teaches the model that
  documents end.
- Memory-mapped rather than loaded. 16 GB is shared between CPU and GPU, and the
  training loop needs that headroom for activations. A memmap lets the OS page cache
  hold the hot part of the corpus and evict the rest under pressure.
- `uint16` on disk. Vocabulary is 8192, so ids fit in 16 bits with room to spare, and
  `uint16` halves the file size and the page cache footprint against `int32`. The
  conversion to int64 happens per batch, on tensors small enough not to matter.

## 3. Shape of the pipeline

Four stages, each independently testable:

```
source files  ->  documents  ->  token ids  ->  shards + manifest
   (fetch)        (iterate)      (encode)        (write)
```

- **fetch** obtains the corpus and verifies it against a recorded digest.
- **iterate** yields documents as `bytes`, one at a time, without holding the corpus
  in memory.
- **encode** turns each document into ids and terminates it with `END_OF_TEXT_ID`.
- **write** appends ids to fixed-size shards and records a manifest.

The seam that matters is between iterate and encode. Everything downstream of
`Iterator[bytes]` is testable with synthetic documents and no download, and that is
where the majority of the test suite will live.

## 4. Streaming, and why the tokenizer cannot do it

`train(corpus: bytes, vocab_size: int)` takes the entire corpus as one `bytes` object
and holds the decoded `str` alongside it during pre-tokenization. That is a documented
limitation carried forward from the tokenizer, and it applies to **training** the
tokenizer, not to using it.

Encoding is different, and this is what makes streaming safe. `encode` is chunk-local:
`pretokenize` splits into chunks that BPE merges never cross, and every chunk re-splits
to itself, both verified by Hypothesis property tests. So encoding a document in
isolation gives exactly the ids that encoding it as part of a larger corpus would give,
provided the split points are respected.

Document boundaries are safe split points, but not for the reason a first reading of
the pattern suggests. `\s*[\r\n]+` is one of the pattern's alternatives, but it is not
the only one that can consume a trailing newline: `pretokenize(b'hello!\n')` gives
`[b'hello', b'!\n']`, with the newline absorbed by the punctuation alternative
(`[^\s\p{L}\p{N}]?...` combined with `[\r\n]*`) rather than isolated by `\s*[\r\n]+`. So
a document boundary is **not always** a chunk boundary in that literal sense.

What actually makes the split safe is narrower, and it does hold: `iter_documents` never
presents two documents' bytes to `pretokenize` at once, so no chunk ever spans one
document's content into the next. Each document is pretokenized as its own isolated
`bytes` object, and `pretokenize` has no lookahead past the end of its input, so nothing
from a neighbouring document can leak into a boundary document's last chunk. That is the
property this pipeline actually depends on, and it holds by construction, independent of
which pattern alternative happens to consume the trailing newline. **Arbitrary byte
offsets are not safe split points** -- cutting mid-chunk (mid-word, say) changes the
tokenization of the bytes on both sides of the cut -- which is why the pipeline streams
documents rather than fixed-size buffers.

One consequence is worth recording. `iter_documents` strips each document's trailing
newline before yielding it, so a document ending in punctuation encodes its last chunk
as, e.g., `!` where pretokenizing the raw, newline-included corpus would have produced
`!\n`. The pipeline never encodes the raw corpus -- only these stripped, per-document
byte strings -- so this is internally self-consistent. But the tokenizer is trained on
corpus text, not on this pipeline's stripped documents. If the tokenizer is trained on
raw corpus bytes while this pipeline encodes document-wise, the merge table will contain
`<punct>\n` merges that encoding here can never fire, wasting vocabulary slots and
skewing the compression comparison against `tiktoken`. Whichever task trains the corpus
tokenizer must train it on this same document-wise convention, or record why not.

## 5. The quadratic encode hazard

`encode` is O(L squared) in chunk length and `pretokenize` imposes no ceiling. Measured
on single-chunk inputs: 2 KB takes 0.085s, 8 KB takes 1.662s, 32 KB takes 22.716s, and
128 KB takes 364.1s. This pipeline is the first caller to hand `encode` real data, so
the hazard becomes live here.

Natural English prose does not reach it. Chunks are words, so they are tens of bytes at
most, and O(L squared) on a 20-byte chunk is free. The danger is a degenerate document:
one long unbroken run of letters or whitespace, which `\p{L}+` or `\s+` will match as a
single chunk. A corpus containing one such run would appear to hang.

**Decision: measure and fail loudly rather than cap.** The pipeline tracks the longest
chunk it has seen and refuses to proceed past a configurable `max_chunk_bytes` ceiling,
raising an error that names the document and the length. Capping instead, by splitting
an over-long chunk, would silently change the tokenization and make the compression
comparison against `tiktoken` meaningless. The observed maximum goes into the manifest,
so the assumption is recorded as a measurement rather than a belief.

### 5.1 Measured chunk length distribution

Measured over the repository's own markdown, Python, and TOML files, since the corpus is
not downloaded yet: 24 files, 155528 bytes, 34789 chunks.

| Percentile | Chunk length |
|---|---|
| p50 | 4 bytes |
| p90 | 9 bytes |
| p99 | 13 bytes |
| p99.9 | 18 bytes |
| max | 54 bytes |

Nothing exceeded 64 bytes. The longest chunk was a 54-byte run of spaces inside a
markdown table in `DESIGN.md`, which is a formatting artifact rather than prose, and it
came from `\s+` matching an unbroken whitespace run. That is worth noting because it is
the same alternative that produces the pathological case: the tail of this distribution
is made of whitespace runs, not words.

Source code is a conservative stand-in. It contains long identifiers, URLs, and aligned
tables that TinyStories prose will not, so the real corpus should be shorter-tailed. The
measurement is replaced with the real one in the plan's gated task, and the ceiling
revised if the stand-in turns out to have been optimistic.

### 5.2 The ceiling

`DEFAULT_MAX_CHUNK_BYTES` is **1024**.

Both bounds are observations rather than preferences. From below, the largest chunk any
real text here produced was 54 bytes, so 1024 leaves roughly 19x headroom and will never
fire on ordinary input. From above, section 5's timings put a 2 KB chunk at 0.085s;
scaling quadratically, a chunk at the ceiling costs about 0.02s, which is affordable as
a rare event and unaffordable as a common one. A ceiling of 4096 would cost roughly 0.34s
per chunk, which is past the point where a handful of them would be noticeable.

The ceiling is a guard against degenerate input, not a tuning parameter. If it ever
fires on real data, the right response is to look at the document, not to raise it.

## 6. Splitting

The split is by **document**, not by token offset. A token-offset split would put the
first half of a story in train and the second half in validation, which leaks.

The procedure is deterministic given a seed:

1. Enumerate documents in source order, assigning each an index.
2. Draw a permutation of the indices from a `numpy.random.default_rng(seed)`.
3. The first `val_fraction` of the permuted order is validation, the rest is training.

Recording the seed and the fraction in the manifest makes the split reproducible from
the manifest alone. TinyStories publishes its own train and validation files; the
pipeline uses a seeded split anyway so that the held-out set is under this project's
control and cannot drift if the upstream files change.

## 7. On-disk format

```
data/
  tinystories/
    manifest.json
    train-00000.bin
    train-00001.bin
    ...
    val-00000.bin
```

Each `.bin` is a headerless flat array of little-endian `uint16`. No header, because the
manifest already carries the metadata and a headerless file is exactly what
`numpy.memmap` wants.

Shards are capped at a fixed token count so that no single file is unwieldy and so that
a failed run can be resumed at shard granularity. The last shard of a split is short.

The manifest records everything needed to reproduce or validate the artifact:

- the tokenizer file's sha256, so a shard set can never be silently paired with the
  wrong tokenizer
- the source files' sha256
- the split seed and validation fraction
- token counts per split, and per shard
- each shard's sha256
- the longest pre-token chunk observed, per section 5
- the nanoscope version that wrote it

## 8. Reading

`ShardedTokens` opens a split's shards as `numpy.memmap` objects and presents them as
one logical sequence, so a caller can ask for the window `[i, i + seq_len)` without
knowing where shard boundaries fall. A window that straddles two shards is served by
concatenating across the seam.

Loading verifies the manifest's tokenizer digest against the tokenizer the caller
holds, and refuses to proceed on a mismatch. Training on shards produced by a different
tokenizer is a silent-garbage failure of exactly the kind this project's serialization
format already guards against.

## 9. CLI

```
nanoscope data prep --source PATH --tokenizer PATH --output DIR
                    [--val-fraction F] [--seed N] [--shard-tokens N]
                    [--max-chunk-bytes N] [--limit N]
```

`--limit` caps the number of documents processed, so a small artifact can be produced
for development and tests without a full run.

Following the tokenizer verb's precedent, every path is validated before any work
begins, since the full run is expensive and a typo should not cost it.

## 10. Testing

Property tests, on synthetic documents, requiring no download:

- Every token id read back from the shards equals what was written, and decoding a
  split's ids reproduces the concatenated documents with `END_OF_TEXT_ID` at the
  boundaries.
- The split is a partition: every document lands in exactly one side, and the same seed
  reproduces the same assignment.
- A window straddling a shard boundary returns the same ids as the same window read
  from an unsharded array.
- A manifest whose tokenizer digest does not match is rejected.
- A document containing an over-long chunk is rejected with the document named.

Only the fetch stage needs the network, and it is the only part that cannot be tested
offline. It is kept as thin as possible for that reason.

## 11. Risks

- **Full-run cost is unmeasured.** Tokenizing the whole corpus is the longest operation
  this project has run so far. The plan will measure it on a `--limit`ed sample and
  extrapolate before committing to a full run.
- **Disk.** Shard size is bounded by the token count, which is not known until the
  tokenizer is trained on this corpus. Both go in the manifest once measured.
- **The tokenizer is not trained yet.** No trained tokenizer artifact exists, so this
  pipeline cannot produce a real artifact until one does. Training it is a prerequisite
  step, not part of this pipeline.

## 12. Questions raised at review, and how they were settled

**1. Should `data prep` train the tokenizer when `--tokenizer` does not exist?**

No. It requires the file to exist and fails otherwise.

Training the tokenizer and preparing the shards are the two most expensive operations in
the project, and folding them into one verb makes them one failure unit: a crash in
preparation would discard a completed training run, which is the same defect the
tokenizer CLI's review round already fixed once. Keeping them separate also keeps the
manifest honest, since it pins a tokenizer by digest and an implicitly trained one would
be an artifact nobody chose or reviewed.

**2. A seeded split, or TinyStories' published train and validation files?**

A seeded split, taken over the published training file.

The deciding argument is that the manifest must be sufficient to reproduce the artifact.
A seed and a fraction are recorded and reproducible; "whatever was in the upstream
validation file on the day we downloaded it" is neither, and it can drift without
notice. The cost is comparability against published TinyStories numbers, which this
project never claims: DESIGN.md's success criteria are internal, comparing ablation
variants against each other on identical data.

The published validation file is left untouched on disk. It costs nothing to keep, and
it gives a genuinely external held-out set if a later question ever needs one.
