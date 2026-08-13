# nanoscope

A transformer built from scratch, for a job-application portfolio. Senior engineers
will read the code and the commit history, and Shivank has to defend every design
decision in an interview.

## Standing authorization: do not stop to ask

Work autonomously. Decide and proceed; do not pause for approval.

This **overrides** the rule in `superpowers:subagent-driven-development` that says a
plan-mandated review finding is the human's decision. It is not. When a review finding
conflicts with what a plan or brief mandates, rule on it yourself, record the ruling and
its reasoning in the SDD ledger, and keep going. The ledger is the audit trail; a
question is not.

Default rulings, so the pattern is predictable:

- A brief's literal code that does not lint, or that makes a claim the code does not
  support, is a defect **in the plan**. Fix the code, note the plan's error.
- A test whose strategy cannot reach the input its name promises is a real finding, even
  when the brief wrote that test. Widen the strategy; never narrow the property.
- A boundary that silently accepts bad input is worth fixing even when nothing currently
  reaches it. Every Important finding on the tokenizer branch was of this shape.

Push and merge without asking as well, once the final whole-branch review is clean.

Still stop for: anything needing a paid API, a large download, or more than about 15
minutes of compute; and anything genuinely destructive or irreversible beyond a normal
push.

## Hard constraints

- **Commits are authored by Shivank Pandey alone.** Never add a `Co-Authored-By`
  trailer or any AI attribution, in commit messages or in documents.
- **No fabricated numbers**, anywhere: code, comments, docs, CLI help text, commit
  messages. Every number traces to something actually observed and rerunnable. Never
  estimate a benchmark.
- **No em dashes in prose.**
- **Never run git write commands from `/Users/shivpanndey05`** or any uninitialised
  subdirectory. The home directory is itself a git repo holding unpushed commits, so a
  git write from the wrong cwd commits to it. All git commands run from this repo root.
- No TODOs or stubs in a final state.

## Environment

Apple M5, 10 cores, 16 GB unified memory, **no CUDA**. MPS only: no flash-attention, no
bf16 autocast parity, flaky `torch.compile`. The binding constraint is unified memory,
not the GPU. There are no LLM API keys on this machine; local Ollama `qwen3:4b` is the
only path, not a fallback.

## Verification

`make check` must be green before any unit of work is done: ruff, `ruff format --check`,
mypy strict, pytest. Run everything through `uv run`.

Two things that have bitten this repo:

- `ruff format` also checks fenced Python blocks **inside markdown**. A doc with a
  ```python fence breaks `make check` unless formatted.
- The whole `RUF` ruleset is selected, so RUF007 rejects `zip(x, x[1:])` in favour of
  `itertools.pairwise`, regardless of what a plan's literal code says.
