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

Document boundaries are safe split points because documents are separated in the source
by a newline, and `\s*[\r\n]+` is one of the pattern's alternatives, so a boundary is
always a chunk boundary. **Arbitrary byte offsets are not safe split points**, which is
why the pipeline streams documents rather than fixed-size buffers.

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

The default ceiling is a decision for the plan, chosen after measuring the actual chunk
length distribution on a sample of the corpus. It is not guessed here.

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

## 12. Open questions for review

1. Should `data prep` train the tokenizer when the `--tokenizer` path does not exist,
   or require it to exist and fail otherwise? Failing is more predictable and keeps the
   two expensive operations separately resumable, which is the current preference.
2. Is a seeded split preferable to TinyStories' published train/validation files? The
   argument for seeding is control and reproducibility; the argument against is
   comparability with published numbers, which this project does not claim.
