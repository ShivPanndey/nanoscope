# nanoscope

A decoder-only transformer built from scratch in PyTorch, used to measure what each modern architectural component (RoPE, GQA, RMSNorm, SwiGLU, weight tying) actually contributes to validation loss and throughput.

**Status: in development.** This README is written last and deliberately contains no results yet. Every number that eventually appears here will come from a script in `src/nanoscope/bench/` with its raw output committed under `results/`, reproducible by a single command.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, the ablation methodology, and the hardware constraints that shaped both.
