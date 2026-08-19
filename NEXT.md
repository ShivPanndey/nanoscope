# Next unit of work

Tokenizer landed 2026-08-11: pre-tokenizer, both trainers, encoder, decoder,
serialization, and the `tokenizer train` CLI verb.

Data pipeline Tasks 1 through 6 landed 2026-08-19, merged to `main` at
`7944d6c` with CI green: the manifest model, document iteration, shard writing
and reading, `prepare` with the seeded document-level split, and the
`data prep` CLI verb. 129 tests. `make check` green.

## Next, in order

1. **Data pipeline Task 7, the real corpus.** The only task left in
   `docs/plans/2026-08-13-data-pipeline.md`, and the one that is **gated**: it
   needs a large download and a run likely to exceed 15 minutes, so it starts
   only on an explicit go-ahead. Everything it depends on is built and merged,
   so it is a measurement exercise rather than a debugging one. It covers the
   TinyStories fetch and sha256, training the real tokenizer artifact, the
   real chunk length distribution against Task 1's stand-in, a `--limit`ed
   throughput probe before the full run, the full preparation, the `tiktoken`
   comparison owed from DESIGN.md section 6, ADR-0005, and rewriting this file.
2. **Model.** Components in dependency order: RMSNorm, RoPE, attention
   (GQA + SDPA), SwiGLU MLP, block, transformer. Every component behind a
   config flag from the start, since retrofitting toggles later is how ablation
   harnesses rot.
3. **Training loop**, then the **calibration run** (200 steps) that sets every
   downstream token budget.

## Carried forward into Task 7

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
- **`encode` is quadratic in chunk length.** Measured on a tokenizer that
  actually merges the repeated byte, single-chunk inputs: 2 KB, 0.085s; 8 KB,
  1.662s; 32 KB, 22.716s; 128 KB, 364.1s. Roughly 16x cost per 4x input.
  `pretokenize` still imposes no ceiling of its own, so the guard lives in the
  pipeline: `prepare` rejects any document containing a chunk longer than
  `DEFAULT_MAX_CHUNK_BYTES` (1024), chosen in Task 1 against the repo's own
  markdown, Python, and TOML as a stand-in corpus. Task 7 re-measures against
  real TinyStories prose and revises the ceiling if the stand-in was optimistic.
- **Two defaults are still unmeasured**, and say so in `prepare.py`:
  `DEFAULT_VAL_FRACTION` (0.01) and `DEFAULT_SHARD_TOKENS` (10,000,000). The
  full run is the first chance to put numbers behind either.
- **Deferred cleanup:** `_squash` is duplicated between the two CLI test files.
  The trigger to extract it is a third CLI verb.

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
- **CLI output assertions must be rerun under `FORCE_COLOR=1`.** Rich
  colourises in CI and not in a plain local run, and a helper that strips
  non-alphanumerics turns an ANSI code into a fragment like `36m` that splits
  words. This broke CI once already.
- A plan's literal code is not trusted to lint. The data pipeline plan hit this
  the same way the tokenizer plan did; run `make check` against it early.
