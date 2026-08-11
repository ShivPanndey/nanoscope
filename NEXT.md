# Next unit of work

Session ended 2026-08-11 at a clean boundary: scaffold complete, `make check` green, nothing half-built.

## Blocked

- **Push to GitHub.** The `gh` token has scopes `gist, read:org, repo` but **not `workflow`**, so pushing `.github/workflows/ci.yml` will be rejected. The refresh attempt expired waiting on browser approval. Rerun:
  ```
  gh auth refresh -h github.com -s workflow
  ```
  Then `gh repo create ShivPanndey/nanoscope --public --source=. --push`.

## Next, in order

1. **Byte-level BPE tokenizer.** Trainer, encoder, decoder. Hypothesis property tests for `decode(encode(s)) == s` over arbitrary Unicode, not just ASCII. Emoji, combining characters, and lone surrogates are where this breaks.
2. **Data pipeline.** TinyStories download, tokenize, memmap shards. Held-out validation split fixed by seed so every ablation sees identical data in identical order.
3. **Model.** Components in dependency order: RMSNorm, RoPE, attention (GQA + SDPA), SwiGLU MLP, block, transformer. Every component behind a config flag from the start, since retrofitting toggles later is how ablation harnesses rot.
4. **Training loop**, then the **calibration run** (200 steps) that sets every downstream token budget.

## Open decision, deferred until calibration produces real numbers

Whether to spend ~25 minutes on one repeat of the baseline at a different seed, to establish a noise floor for the whole ablation table. Multi-seed replication was already declined as too expensive; this is the cheaper version and applies to every row rather than two. Revisit once a run's true cost is measured.

## Standing constraints

- No `Co-Authored-By` trailers. Commits are authored by Shivank Pandey alone.
- Never run git write commands from `/Users/shivpanndey05` or any uninitialized subdirectory: the home directory is itself a repo holding ~40 unpushed commits of unrelated work.
- No fabricated numbers. Calibration measures throughput; it does not get estimated.
