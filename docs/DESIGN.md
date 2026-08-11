# nanoscope: Design

Status: proposed, awaiting approval
Author: Shivank Pandey
Date: 2026-08-11

## 1. Goal

Build a decoder-only transformer language model end to end in PyTorch, with no high-level modeling library, and use it to produce a **measured ablation study** showing what each modern architectural component actually contributes to validation loss and throughput.

The deliverable that matters is the ablation table, not the model. A 12.6M parameter model trained on TinyStories will generate simple, recognizable children's-story prose. It will not be an impressive chatbot, and the README will say so directly rather than implying otherwise.

### Non-goals

- Competing with any published model on quality. Out of scope at this scale.
- Distributed or multi-GPU training. There is one GPU and it is integrated.
- Serving or deployment. That is Project 4 (`inferno`), which will consume this model's checkpoint.
- Novel research. Every component here is known. The contribution is rigorous measurement on accessible hardware.

## 2. Hardware reality

All design decisions below follow from this. Apple M5, 10 CPU cores, 10 GPU cores, **16 GB unified memory, no CUDA**.

| Standard practice | Why it fails here | What I do instead |
|---|---|---|
| FlashAttention | CUDA-only kernel | `F.scaled_dot_product_attention` (Metal fast path) as the tuned path, plus a hand-written naive attention as the ablation baseline |
| bf16 autocast | No bf16 parity on MPS | fp16 autocast, with an fp32 path measured rather than assumed. If fp16 destabilizes training, that is a reportable result, not a failure to hide |
| `GradScaler` | CUDA-coupled in PyTorch | Manual loss scaling if fp16 needs it, otherwise omit and say why |
| `torch.compile` | Incomplete and flaky on MPS | Config flag, benchmarked. If it breaks or does not help, the README reports that with numbers |
| Large batches | 16 GB is shared between CPU and GPU | Batch 32 at seq 512, with gradient accumulation to reach the effective batch size |

**Known trap I am explicitly guarding against:** MPS silently falls back to CPU for unimplemented ops, which looks like a mysterious 10x slowdown. The benchmark harness will run with `PYTORCH_ENABLE_MPS_FALLBACK=0` so any unsupported op raises loudly instead of quietly destroying throughput.

## 3. Model architecture

Config-driven, every component independently toggleable, because the ablation study depends on it.

| Parameter | Value |
|---|---|
| Layers | 6 |
| `d_model` | 384 |
| Query heads | 6 (head_dim 64) |
| KV heads (GQA) | 2 |
| MLP hidden (SwiGLU) | 1024 |
| Context length | 512 |
| Vocab | 8192 |

Parameter count, computed rather than guessed:

```
embedding (tied with output head)  8192 x 384          =  3,145,728
per layer:
  q_proj  384x384 = 147,456
  k_proj  384x128 =  49,152     (2 KV heads x 64)
  v_proj  384x128 =  49,152
  o_proj  384x384 = 147,456
  swiglu  3 x (384x1024)        =  1,179,648
  2 x RMSNorm                    =        768
  layer total                    =  1,573,632
6 layers                                             =  9,441,792
final norm                                           =        384
                                                       ----------
total                                                 = 12,587,904   (~12.6M)
non-embedding                                          =  9,442,176
```

Toggleable components: RoPE (vs learned absolute positions), GQA (vs full MHA), RMSNorm (vs LayerNorm), SwiGLU (vs ReLU MLP), weight tying (vs untied head).

## 4. The ablation study, and the trap in it

This is the centerpiece, so the methodology needs to be defensible in an interview.

**The confound:** naively toggling a component changes the parameter count, so any loss difference conflates "this component is better" with "this model is bigger." Removing SwiGLU for a standard ReLU MLP is the worst case: SwiGLU uses three matrices where ReLU MLP uses two, so at equal hidden width the ReLU variant has ~2.4M fewer parameters and would look worse for the wrong reason.

**The control:** every ablation is **parameter-matched** to within 1% of baseline by adjusting MLP hidden width, and the doc reports the actual parameter count of every variant alongside its loss. Where exact matching is impossible, I report the mismatch rather than hiding it. Untying weights genuinely adds 3.1M parameters and cannot be matched, so that row is reported as the confounded comparison it is, with the caveat stated.

**Fairness rules for every run:** identical data order (fixed seed), identical token budget, identical LR schedule, identical hardware and thermal conditions (runs are serialized, never run concurrently). Each variant is trained from scratch.

**Metrics per variant:** final validation loss, validation loss curve, training tokens/sec, peak memory, parameter count.

**Honest limitation I will state up front:** at 12.6M parameters and **one seed per variant** (decided: multi-seed replication is out of scope for the compute budget), small loss differences are not statistically meaningful. The README will report the seed count explicitly and will use language calibrated to that evidence: "directionally consistent with" rather than "proves." Where two variants land close together I will say they are not separated at this sample size rather than ranking them.

## 5. Calibration first, then budgets

I do not yet know this chip's training throughput, and the no-fabrication rule means I will not estimate it in the doc. So step one after scaffolding is a **calibration run**: train the baseline for 200 steps and measure actual tokens/sec and peak memory.

Every downstream budget is derived from that measurement:
- Per-ablation token budget set so each run lands near 25 minutes.
- Total ablation cost = 6 variants x that budget, reported before I start the sweep.
- A single longer baseline run for the released checkpoint and generation samples.

If calibration shows the 3 to 5 hour estimate was wrong in either direction, I bring the real number back to you before running the sweep.

## 6. Tokenizer

Byte-level BPE from scratch, vocab 8192, trained on the TinyStories training split.

- Byte-level so there are no out-of-vocabulary inputs by construction.
- Correctness bar: round-trip fidelity, `decode(encode(s)) == s`, enforced with Hypothesis property tests over arbitrary Unicode, not just the ASCII happy path. Emoji, combining characters, and lone surrogates are where naive implementations break.
- Reported comparison against `tiktoken`: compression ratio (bytes per token) on held-out text, plus encode throughput. I expect `tiktoken` to win on speed by a wide margin since it is Rust, and saying so is more credible than pretending otherwise.

## 7. Interfaces

```
nanoscope/
  src/nanoscope/
    config.py        typed config (Pydantic), every architectural toggle
    tokenizer/       BPE trainer, encoder, decoder
    model/           attention, mlp, norms, rope, block, transformer
    data/            TinyStories download, tokenize, memmap shards
    train/           loop, schedule, checkpointing, resume, logging
    infer/           kv cache, greedy/temperature/top-k/top-p sampling
    bench/           throughput, memory, ablation runner
  tests/
  results/           raw benchmark JSON, committed
  docs/              DESIGN, ARCHITECTURE, DECISIONS
```

CLI (`nanoscope <verb>`): `tokenizer train`, `data prep`, `train`, `generate`, `bench`, `ablate`.

Config is a typed object with no magic numbers in code. Every experiment is fully described by its config file, which is committed alongside its results so any number in the README can be reproduced by one command.

## 8. Training loop

Gradient accumulation, cosine schedule with linear warmup, gradient clipping at 1.0, mixed precision, checkpoint and resume, and loss plus throughput logging to JSONL.

**Resume correctness is a real test, not an afterthought.** The test saves a checkpoint mid-run, resumes, and asserts the loss curve and RNG state continue identically to an uninterrupted run. Most hobby implementations get this wrong by forgetting optimizer state or the data loader position, and the test is what proves this one does not.

## 9. Inference

KV cache with pre-allocated tensors, and sampling implemented from scratch (greedy, temperature, top-k, top-p).

Correctness bar: cached and uncached generation must produce **identical** token sequences at temperature 0. This is the test that catches cache indexing bugs, which are otherwise invisible because the output still looks like plausible text.

Benchmark: tokens/sec at batch 1, 8, 32, with and without cache, plus peak memory.

## 10. Risks

| Risk | Mitigation |
|---|---|
| fp16 instability on MPS (NaN loss) | Detect NaN and abort early; fall back to fp32 and report the finding rather than quietly switching |
| Silent MPS-to-CPU op fallback | `PYTORCH_ENABLE_MPS_FALLBACK=0` in benchmarks so it fails loudly |
| Ablation differences within noise | Parameter-match, fix seeds, report noise honestly, add seeds where it matters |
| 16 GB pressure at batch 32 | Calibration run measures peak memory before committing to the sweep |
| Thermal throttling skewing throughput | Serialize runs, record wall-clock and ambient conditions, re-run the baseline last as a drift check |
| Scope creep | Ablation table and architecture doc are the deliverable; sample quality is explicitly not |

## 11. Success criteria

1. Model trains to a converging validation loss on TinyStories.
2. Ablation table with measured loss and throughput for all toggles, parameter-matched, reproducible by one command.
3. Cached and uncached generation provably identical at temperature 0.
4. Tokenizer round-trips arbitrary Unicode under property tests.
5. Resume produces a bit-identical continuation.
6. CI green: ruff, mypy strict, pytest.
7. Every number in the README traceable to a script in `bench/` and raw output in `results/`.

## 12. Decisions

Resolved at review on 2026-08-11:

1. **Seeds:** one seed per variant. Multi-seed replication rejected as not worth the compute. Consequence, accepted deliberately: the ablation table shows a measured direction, not a statistically established one, and the prose must be worded to match.
2. **Released checkpoint:** the best-performing configuration, not the parameter-matched baseline. The ablation table remains parameter-matched for fairness; the shipped model is simply the best one found.
3. **Architecture:** approved as specified above.

Still open, resolved by measurement rather than opinion:

4. Token budget per ablation run, set from the calibration run in section 5.
