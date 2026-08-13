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
- **The `tiktoken` comparison from DESIGN.md section 6**: compression ratio in
  bytes per token, plus encode throughput. Needs the held-out validation split
  to be meaningful. Add `tiktoken` to the dev group then.
- **Corpus memory.** `train` takes the whole corpus as `bytes` and holds the
  decoded `str` alongside it during pre-tokenization. If the TinyStories split
  is too large for 16 GB, train on a seeded, documented subsample and record
  its sha256. Do not switch to a streaming trainer: chunking at arbitrary
  boundaries splits pre-tokens and perturbs the pair counts. Decided by
  measurement, not now.
- **`encode` is quadratic in chunk length, and `pretokenize` imposes no
  ceiling on chunk length.** Measured on a tokenizer that actually merges the
  repeated byte, single-chunk inputs: 2 KB, 0.085s; 8 KB, 1.662s; 32 KB,
  22.716s; 128 KB, 364.1s. Roughly 16x cost per 4x input, i.e. quadratic. The
  design spec's premise that "chunks are short by construction, so the
  quadratic inner loop is bounded" (section 5.3) does not hold: `\p{L}+`,
  `\s+`, and `[^\s\p{L}\p{N}]+` all match unbounded runs. No shipped path
  reaches this today (`tokenizer train` never calls `encode`); the data
  pipeline is the first that will, since it calls `encode` on the entire
  corpus.

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
- This plan's own literal code did not lint as written in three separate
  tasks: the mandated `zip(x, x[1:])` pairing idiom loses to `itertools.pairwise`
  under this repo's RUF007 rule, and tasks 2, 3, and 4 each substituted
  `pairwise` under a controller ruling. Run `make check` against a plan's
  literal code early rather than trusting it to lint clean.
