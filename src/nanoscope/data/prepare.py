"""`prepare`: turns a corpus file and a trained tokenizer into a
seed-reproducible train/val shard set plus manifest.

This is where the four pipeline stages of design spec section 3 meet:
`iter_documents` (Task 3) iterates, `Tokenizer.encode` (already merged)
encodes, and `ShardWriter` (Task 4) writes. This module adds the two pieces
nothing downstream owns yet: the document-level train/val split (section 6)
and the quadratic-encode guard (section 5).

**`source` is read three times, not once.** `sha256_file(source)` reads the whole
file once to compute its digest. Assigning document `i` to train or val requires
knowing the total document count `N` up front, since the split is drawn as one
permutation of `range(N)` (section 6) rather than a per-document coin flip;
`iter_documents` is a generator that never holds the corpus in memory (Task 3's
whole point), so the cheapest way to learn `N` without giving that up is a second
read that only counts. The third read is the real work: encoding every document
and writing it to whichever split it was assigned. This pipeline has no
distributed or streaming ambition (design spec section 1), so reading one file
three times is a clean trade against holding it, or a document list built from
it, in memory.
"""

from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import numpy as np

import nanoscope
from nanoscope.data.documents import iter_documents
from nanoscope.data.manifest import Manifest, sha256_file
from nanoscope.data.shards import MAX_TOKEN_ID, ShardWriter
from nanoscope.tokenizer import Tokenizer
from nanoscope.tokenizer.pretokenize import pretokenize
from nanoscope.tokenizer.vocab import END_OF_TEXT_ID

# Design spec section 5.2: bounded below by the largest chunk any real text
# measured so far produced (54 bytes, section 5.1's stand-in corpus), and
# above by the quadratic cost of `encode` (a chunk at this ceiling costs
# about 0.02s; 4096 would cost about 0.34s). A guard against degenerate
# input, not a tuning parameter -- see section 5.2 for the full reasoning.
DEFAULT_MAX_CHUNK_BYTES = 1024

# No measurement backs this default yet -- unlike DEFAULT_MAX_CHUNK_BYTES
# above, no real TinyStories document count exists until the gated corpus
# task runs it. 1% is a conventional small held-out fraction for a
# language-modeling split, chosen rather than derived. Revisit once that
# task measures how many documents 1% actually holds out.
DEFAULT_VAL_FRACTION = 0.01

# Also unmeasured. 10,000,000 `uint16` tokens is about 19 MiB per shard on
# disk, a value chosen to trade the blast radius of one corrupt or truncated
# shard against file count (larger shards mean fewer files for
# `ShardedTokens.open` to `mmap`). Revisit once the gated corpus task
# measures the real token count this pipeline will actually write, and set
# this from that measurement.
#
# An earlier version of this comment justified the value by resume
# granularity, "a failed run resumes at the last complete shard". There is
# no resume path: `prepare` always starts from `_shard_index = 0` and
# rewrites every shard, and nothing reads an existing manifest to continue
# from it. The rationale is stated in terms of what the code does.
DEFAULT_SHARD_TOKENS = 10_000_000


def _limited(documents: Iterable[bytes], limit: int | None) -> Iterable[bytes]:
    return documents if limit is None else islice(documents, limit)


def _count_documents(source: Path, limit: int | None) -> int:
    return sum(1 for _ in _limited(iter_documents(source), limit))


def _validation_mask(document_count: int, val_fraction: float, seed: int) -> set[int]:
    """The set of document indices assigned to validation, per design spec
    section 6: permute `range(document_count)` with `default_rng(seed)`, and
    the first `val_fraction` of the permuted order is validation.

    `round` rather than `floor` or `ceil`: neither is specified, and `round`
    is the choice that treats `val_fraction` as the fraction it's named for
    rather than always rounding the split toward one side. But `round` alone
    lets a strictly positive `val_fraction` round down to zero validation
    documents -- `round(100 * 0.005) == 0`, and Python's banker's rounding
    makes it worse (`round(1 * 0.5) == 0` too) -- which would silently hand
    back an empty validation split for a fraction the caller believed was
    nonzero. A caller who asked for `val_fraction == 0.0` gets exactly that;
    anyone who asked for more than zero gets at least one document, even if
    that biases a very small or very lopsided split up by one. Clamped to
    `[0, document_count]` throughout, both as this floor's ceiling and as a
    defensive measure against float error at the `val_fraction == 1.0` edge.
    """
    order = np.random.default_rng(seed).permutation(document_count)
    val_count = round(document_count * val_fraction)
    if val_fraction > 0.0 and document_count > 0:
        val_count = max(1, val_count)
    val_count = max(0, min(document_count, val_count))
    return set(order[:val_count].tolist())


def prepare(
    source: Path,
    tokenizer_path: Path,
    output_dir: Path,
    *,
    seed: int,
    val_fraction: float,
    shard_tokens: int,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    limit: int | None = None,
) -> Manifest:
    """Tokenize every document in `source` with the tokenizer at
    `tokenizer_path`, split them by document into train and val, write both
    splits' shards under `output_dir`, and return the `Manifest` describing
    the result (also written to `output_dir / "manifest.json"`).

    `output_dir` is created (including parents) if it does not already
    exist. `limit`, if given, caps the number of documents processed --
    useful for a small artifact in development and tests, matching design
    spec section 9's `--limit`. `limit` and `max_chunk_bytes` are both
    recorded on the returned manifest: the split depends on how many
    documents were actually processed, not on how many `source` contains, so
    without `limit` on the manifest a `--limit`ed run's split would not be
    reproducible from the manifest alone, contrary to design spec section 6.

    Each document is encoded with `tokenizer.encode` and terminated with
    `END_OF_TEXT_ID`. Every document's pre-token chunks (`pretokenize`,
    the same split `encode` uses internally) are measured *before* that
    document is handed to `encode`: `encode` is O(L^2) in chunk length
    (design spec section 5), so checking chunk lengths first means a
    degenerate document is rejected before the quadratic cost it would
    otherwise trigger, not after. Raises `ValueError` naming the document's
    index and the observed length if any chunk exceeds `max_chunk_bytes`.
    The longest chunk actually observed is recorded on the returned
    manifest as `max_chunk_bytes_observed`.

    Raises `ValueError` if `val_fraction` is outside `[0, 1]`, if the
    tokenizer's vocabulary does not fit the `uint16` on-disk format, or if
    `source` holds no documents. Raises whatever `Tokenizer.load` raises for
    a malformed tokenizer file.
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0, 1], got {val_fraction}")
    # The CLI already bounds `--limit` at 1, but `prepare` is importable and
    # `limit=0` would otherwise be reported below as an empty corpus, naming
    # the source for a fault in the argument.
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")

    tokenizer = Tokenizer.load(tokenizer_path)
    # Checked here, up front, rather than left to `ShardWriter.write`'s
    # per-id range check. That check is correct but it fires on the first
    # document that happens to merge into a high enough rank, which on a
    # multi-gigabyte corpus can be hours in and after shards are on disk.
    # The vocabulary size is known before a single byte is encoded, and
    # `tokenizer train` puts no ceiling on `--vocab-size`, so an oversized
    # tokenizer is a reachable input rather than a hypothetical one.
    if tokenizer.vocab_size > MAX_TOKEN_ID + 1:
        raise ValueError(
            f"tokenizer at {tokenizer_path} has a vocabulary of {tokenizer.vocab_size}, "
            f"which does not fit the uint16 shard format (at most {MAX_TOKEN_ID + 1})"
        )
    tokenizer_sha256 = sha256_file(tokenizer_path)
    source_sha256 = sha256_file(source)

    document_count = _count_documents(source, limit)
    # Before `mkdir`, so a mistyped `--source` does not leave an output
    # directory behind. An empty corpus otherwise succeeds: the loop never
    # runs, both writers close empty, and a manifest with no shards is
    # written, so a truncated or misnamed source is indistinguishable from a
    # real run until the training split turns out to be empty much later.
    # `limit` cannot cause this on its own: the CLI bounds it at 1, so any
    # non-empty corpus yields at least one document. An empty count means an
    # empty source.
    if document_count == 0:
        raise ValueError(f"{source} contains no documents")

    output_dir.mkdir(parents=True, exist_ok=True)

    val_indices = _validation_mask(document_count, val_fraction, seed)

    train_writer = ShardWriter(output_dir, "train", shard_tokens)
    val_writer = ShardWriter(output_dir, "val", shard_tokens)
    max_chunk_bytes_observed = 0

    # Both writers must be closed on the way out even when the loop below
    # raises partway through (an over-long chunk) -- otherwise the shard file
    # handle open at the moment of the raise is never closed and survives
    # until GC. The already-written shard files this leaves behind are
    # expected debris (see the module docstring); an unclosed file handle is
    # not. `train_entries`/`val_entries` are only used below the `try`, which
    # is only reached if no exception propagated, so both are always the real
    # `close()` result by the time they are read.
    try:
        for index, document in enumerate(_limited(iter_documents(source), limit)):
            chunks = pretokenize(document)
            longest = max((len(chunk) for chunk in chunks), default=0)
            if longest > max_chunk_bytes:
                raise ValueError(
                    f"document {index} contains a pre-token chunk of {longest} bytes, "
                    f"exceeding max_chunk_bytes={max_chunk_bytes}"
                )
            max_chunk_bytes_observed = max(max_chunk_bytes_observed, longest)

            # `encode_chunks`, not `encode`: `encode` would run the same
            # regex pass over the same bytes a second time, and the chunks
            # the ceiling was just measured against are exactly the ones it
            # would produce. Measured at about 42% of tokenization wall time
            # on a stand-in corpus (see `Tokenizer.encode_chunks`).
            ids = tokenizer.encode_chunks(chunks)
            ids.append(END_OF_TEXT_ID)
            writer = val_writer if index in val_indices else train_writer
            writer.write(ids)
    finally:
        # Nested so the second close runs even if the first raises. Both do
        # real I/O -- `close()` reopens each finished shard to digest it --
        # so a full disk or a revoked mount can fail one of them, and without
        # the nesting a raising `train_writer.close()` leaks the val handle,
        # which is the exact leak this block exists to prevent.
        #
        # A close that raises still replaces an in-flight exception from the
        # loop; that is what a `finally` does, and the original survives as
        # the new exception's `__context__` rather than being lost. Not worth
        # the machinery to invert: a failure to close is itself a failure the
        # caller must see, and both are on the traceback.
        try:
            train_entries = train_writer.close()
        finally:
            val_entries = val_writer.close()

    manifest = Manifest(
        tokenizer_sha256=tokenizer_sha256,
        source_sha256=source_sha256,
        seed=seed,
        val_fraction=val_fraction,
        limit=limit,
        max_chunk_bytes=max_chunk_bytes,
        max_chunk_bytes_observed=max_chunk_bytes_observed,
        nanoscope_version=nanoscope.__version__,
        shards=train_entries + val_entries,
    )
    manifest.save(output_dir / "manifest.json")
    return manifest
