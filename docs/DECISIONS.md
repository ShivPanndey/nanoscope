# Decision log

ADR-style entries, appended as each component lands. Each records what was built, why that approach, what else was considered, and what the trade-off was.

---

## ADR-0001: uv and Python 3.13 for the toolchain

**Date:** 2026-08-11
**Status:** accepted

**Context.** The machine had only system Python 3.9 and a Homebrew 3.13, with no environment manager. The project needs reproducible dependency resolution because benchmark numbers are only meaningful if the environment that produced them can be recreated.

**Decision.** Use `uv` with a committed `uv.lock` and `requires-python = ">=3.13"`.

**Alternatives considered.**
- *Homebrew Python plus venv and `requirements.txt`.* Works, but pip's resolver does not produce a lockfile with hashes by default, so "pinned" would have meant pinned direct dependencies and floating transitive ones. For a repo whose central claim is reproducible measurement, that is a weak foundation.
- *Poetry.* Mature and produces a real lock. Slower, and its resolver has historically struggled with torch's platform-specific wheels.
- *conda.* Standard in ML and handles non-Python dependencies well, but heavyweight and its environment files are less reproducible across platforms than a hash-pinned lock.

**Trade-off.** `uv` is young enough that a reviewer may not have used it. Mitigated by the `Makefile`, which means nobody needs to know `uv` to run `make check`.

---

## ADR-0002: Parameter-matched ablations

**Date:** 2026-08-11
**Status:** accepted

**Context.** The ablation table is the centerpiece of this repo. The obvious implementation, toggling a component off and retraining, is subtly invalid: several of these components change the parameter count. Removing SwiGLU in favor of a standard ReLU MLP drops roughly 2.4M parameters at equal hidden width, because SwiGLU uses three projection matrices where a ReLU MLP uses two. A naive table would then show SwiGLU "winning" partly because the SwiGLU model is simply bigger.

**Decision.** Every ablation variant is parameter-matched to the baseline within 1% by adjusting MLP hidden width, and every row of the table reports its actual parameter count next to its loss. Weight tying is the exception: untying genuinely adds a 3.1M parameter output head and cannot be matched without changing something else material. That row is published and explicitly labeled as confounded rather than presented as a clean comparison.

**Alternatives considered.**
- *Ignore it and toggle naively.* Simplest, and it is what most hobby implementations do. Rejected because the resulting table would be measuring model size while claiming to measure architecture, which is the exact failure mode this repo is meant to demonstrate awareness of.
- *Match on FLOPs instead of parameters.* Arguably more principled for throughput claims, since parameter count and compute cost diverge. Rejected as the primary control because it is harder to verify by inspection, but throughput is reported per variant anyway, so a reader can see the compute side directly.
- *Match by scaling depth rather than MLP width.* Rejected because changing depth alters the optimization problem (gradient path length) far more than widening an MLP does, introducing a worse confound than the one being fixed.

**Trade-off.** Parameter matching means each variant is not the textbook configuration of that component, so the numbers answer "what does this component contribute at fixed budget" rather than "what does the standard recipe score." That is the more useful question, but it needs stating plainly in the README so nobody misreads the table.

---

## ADR-0003: One seed per variant

**Date:** 2026-08-11
**Status:** accepted, with a known limitation

**Context.** With one training run per variant there is no estimate of run-to-run variance, so it is impossible to say whether a small loss gap is a real effect or noise.

**Decision.** One seed per variant. Multi-seed replication was considered and rejected against the compute budget.

**Alternatives considered.**
- *Three seeds on the two closest variants (~1 hour).* Rejected as too expensive for the marginal claim it buys.
- *One repeat run of the baseline config at a different seed (~25 minutes) to establish a noise floor for the whole table.* Proposed and still open pending calibration numbers. Cheaper than the above and applies to every row rather than two, so it is the better value if any replication budget is available at all.

**Trade-off.** The README must be worded to match the evidence: the table shows a measured direction, not a statistically established one. Language like "proves" or "significantly better" would be unsupported and is therefore banned in this repo's prose.

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

**Trade-off.** Writing two trainers costs real code: `train.py` holds both
implementations plus their shared helpers in 141 lines, and the differential
test only proves they agree, not that either is optimal. That is the right
guarantee to buy: BPE has no contested definition of optimal, but it does have
a well-known failure mode in incremental bookkeeping, and this closes it.
