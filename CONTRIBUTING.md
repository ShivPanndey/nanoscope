# Contributing

## Setup

```bash
make setup      # uv sync --all-groups
make check      # lint, typecheck, tests: everything CI runs
```

`make check` runs exactly what CI runs, so a green local check means a green pipeline.

## Standards

- **Types.** Full annotations, `mypy` in strict mode, no `Any` escapes without a comment explaining why.
- **Lint.** `ruff` for both linting and formatting. No suppressions without a justifying comment.
- **Tests.** Meaningful tests, not coverage padding. Property-based tests with Hypothesis where the input space is large enough to hide edge cases, which for this repo means the tokenizer above all.
- **No stubs.** No `TODO`, no `NotImplementedError`, no placeholder functions on `main`. A CLI verb is registered only once the thing behind it works.

## Test markers

| Marker | Meaning |
|---|---|
| `slow` | Trains or benchmarks. Excluded from the default run |
| `mps` | Requires Apple Metal. Skipped in CI, which is CPU-only |

CI runs `-m "not slow and not mps"`. Metal-specific behavior is verified locally, and any claim that depends on it is backed by committed output in `results/` rather than by CI.

## Benchmarks

Benchmarks run with `PYTORCH_ENABLE_MPS_FALLBACK=0`. This is deliberate: MPS silently falls back to CPU for unimplemented operations, which presents as an unexplained slowdown rather than an error. Failing loudly is the only way throughput numbers can be trusted.

## Claims and evidence

Any number appearing in a README, docstring, or chart must be produced by a committed script and have its raw output committed under `results/`. If a measurement cannot be run in a given environment, that is stated rather than estimated.

Because the ablations use a single seed per variant, prose describing results uses language calibrated to that evidence. "Directionally consistent with" is supportable. "Proves" is not.

## Commits

Conventional commits (`feat:`, `fix:`, `test:`, `perf:`, `docs:`, `refactor:`). Explain why, not just what. Commit at each logically complete unit rather than in large batches.
