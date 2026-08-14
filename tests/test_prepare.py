"""Tests for `prepare`: the pipeline entry point that turns a corpus file and
a tokenizer into a seed-reproducible train/val shard set plus manifest.

Design spec section 6: the split is by document, not by token offset --
enumerate documents, permute the indices with `numpy.random.default_rng(seed)`,
and the first `val_fraction` of the permuted order is validation. Section 5:
the pipeline tracks the longest pre-token chunk seen and refuses to proceed
past `max_chunk_bytes`, naming the document and the observed length, before
the quadratic encode hazard can fire.

All tests use synthetic documents and a small hand-built tokenizer; nothing
here downloads or trains anything.
"""

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import nanoscope
from nanoscope.data import prepare
from nanoscope.data.manifest import Manifest, sha256_file
from nanoscope.data.shards import ShardedTokens
from nanoscope.tokenizer import Tokenizer
from nanoscope.tokenizer.vocab import END_OF_TEXT_ID


def _write_tokenizer(path: Path) -> Tokenizer:
    """A small, genuinely non-trivial hand-built tokenizer: one real merge
    (the byte pair for 'd' and 'o', both of which appear in this file's
    `doc-N` test documents), not just the identity byte-level vocabulary."""
    tokenizer = Tokenizer(merges=[(ord("d"), ord("o"))])
    tokenizer.save(path)
    return tokenizer


def _write_corpus(path: Path, docs: list[bytes]) -> None:
    path.write_bytes(b"".join(doc + b"\n" for doc in docs))


def _decode_split(tokenizer: Tokenizer, reader: ShardedTokens) -> list[bytes]:
    """Read every id in `reader`, split on `END_OF_TEXT_ID`, and decode each
    piece back to the document bytes it came from.

    Asserts there is no dangling piece after the last terminator, i.e. that
    `END_OF_TEXT_ID` really does sit at every document boundary and nowhere
    else leaves ids unterminated.
    """
    ids = reader.window(0, len(reader)).tolist()
    docs: list[bytes] = []
    current: list[int] = []
    for token_id in ids:
        if token_id == END_OF_TEXT_ID:
            docs.append(tokenizer.decode(current))
            current = []
        else:
            current.append(token_id)
    assert current == [], "ids remain after the last END_OF_TEXT_ID"
    return docs


# ---------------------------------------------------------------------------
# The split is a partition
# ---------------------------------------------------------------------------


@given(
    document_count=st.integers(min_value=0, max_value=40),
    val_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None)
def test_the_split_is_a_partition_of_every_document(
    document_count: int, val_fraction: float, seed: int
) -> None:
    """Every document index lands on exactly one side, and the union of the
    two sides' decoded documents is exactly the original set -- over a range
    of document counts and fractions, not just one hand-picked example.

    Built with `tempfile.TemporaryDirectory()` rather than the `tmp_path`
    fixture: `tmp_path` is function-scoped and Hypothesis reruns this body
    many times per test function, so a fixture-backed path would alias
    across examples (the same lesson Task 3 and Task 4 already applied).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer = _write_tokenizer(tokenizer_path)
        docs = [f"doc-{i}".encode() for i in range(document_count)]
        source = tmp_path / "corpus.txt"
        _write_corpus(source, docs)
        output_dir = tmp_path / "out"

        manifest = prepare(
            source=source,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            seed=seed,
            val_fraction=val_fraction,
            shard_tokens=1_000_000,
        )

        train_reader = ShardedTokens._from_entries(output_dir, manifest.shards, "train")
        val_reader = ShardedTokens._from_entries(output_dir, manifest.shards, "val")
        train_docs = _decode_split(tokenizer, train_reader)
        val_docs = _decode_split(tokenizer, val_reader)

        assert set(train_docs) | set(val_docs) == set(docs)
        assert set(train_docs).isdisjoint(val_docs)
        assert len(train_docs) + len(val_docs) == document_count


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_reproduces_the_same_shard_content(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(20)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)

    def _run(output_dir: Path) -> Manifest:
        return prepare(
            source=source,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            seed=42,
            val_fraction=0.3,
            shard_tokens=1_000_000,
        )

    def _signature(manifest: Manifest) -> list[tuple[str, int, str]]:
        return sorted((entry.split, entry.tokens, entry.sha256) for entry in manifest.shards)

    manifest_a = _run(tmp_path / "out-a")
    manifest_b = _run(tmp_path / "out-b")
    assert _signature(manifest_a) == _signature(manifest_b)


def test_different_seeds_generally_produce_different_assignments(tmp_path: Path) -> None:
    """The word "generally" cannot mean "this one specific other seed differs," since
    that risks flaking on whichever pair happens to collide. Instead: run
    ten different seeds and require at least two distinct outcomes among
    them. With 30 documents split roughly half and half, there are
    C(30, 15) ~= 1.55e8 possible validation subsets, so the chance that ten
    independently seeded permutations all land on the exact same one is not
    a realistic flake risk, while the assertion itself pins to no particular
    seed pair.
    """
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(30)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)

    signatures: set[frozenset[bytes]] = set()
    for seed in range(10):
        output_dir = tmp_path / f"out-{seed}"
        manifest = prepare(
            source=source,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            seed=seed,
            val_fraction=0.5,
            shard_tokens=1_000_000,
        )
        val_reader = ShardedTokens._from_entries(output_dir, manifest.shards, "val")
        signatures.add(frozenset(_decode_split(tokenizer, val_reader)))

    assert len(signatures) > 1


# ---------------------------------------------------------------------------
# Decoding reproduces documents, with END_OF_TEXT_ID at each boundary
# ---------------------------------------------------------------------------


def test_decoding_a_splits_shards_reproduces_its_documents_in_source_order(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(16)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    seed = 7
    val_fraction = 0.25
    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=seed,
        val_fraction=val_fraction,
        shard_tokens=1_000_000,
    )

    # Independently recompute expected split membership per design spec
    # section 6, rather than trusting prepare()'s own internals: permute the
    # indices with `default_rng(seed)`; the first `val_fraction` of the
    # permuted order is validation.
    order = np.random.default_rng(seed).permutation(len(docs))
    val_count = round(len(docs) * val_fraction)
    val_indices = set(order[:val_count].tolist())
    train_indices = set(range(len(docs))) - val_indices

    splits: list[tuple[Literal["train", "val"], set[int]]] = [
        ("train", train_indices),
        ("val", val_indices),
    ]
    for split, indices in splits:
        reader = ShardedTokens._from_entries(output_dir, manifest.shards, split)
        ids = reader.window(0, len(reader)).tolist()
        assert ids.count(END_OF_TEXT_ID) == len(indices)
        decoded = _decode_split(tokenizer, reader)
        assert decoded == [docs[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# The chunk ceiling
# ---------------------------------------------------------------------------


def test_a_document_with_an_overlong_chunk_is_rejected_and_named(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    # Document 2's `pretokenize` chunk is one 20-byte run of letters, well
    # past the 10-byte ceiling below. The others are all well under it.
    docs = [b"short", b"also-short", b"a" * 20, b"more"]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match=r"document 2\b") as exc_info:
        prepare(
            source=source,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            seed=0,
            val_fraction=0.0,
            shard_tokens=1_000_000,
            max_chunk_bytes=10,
        )
    assert "20" in str(exc_info.value)


def test_max_chunk_bytes_observed_is_recorded_on_the_manifest(tmp_path: Path) -> None:
    """The longest chunk sits in the *middle* document, not the last one, so
    a running-max implementation and a bug that only records the last
    document's max would disagree: the former reports 7 (from the middle
    document's "-longer" chunk), the latter would report 4 (the last
    document's "tiny", the whole of one chunk).
    """
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [b"short", b"a-much-longer-chunk-here", b"tiny"]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=0,
        val_fraction=0.0,
        shard_tokens=1_000_000,
        max_chunk_bytes=999,
    )
    assert manifest.max_chunk_bytes_observed == 7  # the "-longer" chunk, middle document


# ---------------------------------------------------------------------------
# val_fraction at the edges
# ---------------------------------------------------------------------------


def test_val_fraction_zero_puts_everything_in_train_with_no_val_shards(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(5)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=0,
        val_fraction=0.0,
        shard_tokens=1_000_000,
    )

    assert manifest.shards
    assert all(entry.split == "train" for entry in manifest.shards)
    val_reader = ShardedTokens._from_entries(output_dir, manifest.shards, "val")
    assert len(val_reader) == 0
    assert val_reader.window(0, 0).shape == (0,)


def test_val_fraction_one_puts_everything_in_val_with_no_train_shards(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(5)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=0,
        val_fraction=1.0,
        shard_tokens=1_000_000,
    )

    assert manifest.shards
    assert all(entry.split == "val" for entry in manifest.shards)
    train_reader = ShardedTokens._from_entries(output_dir, manifest.shards, "train")
    assert len(train_reader) == 0
    assert train_reader.window(0, 0).shape == (0,)


def test_a_small_positive_val_fraction_still_gets_at_least_one_val_document(
    tmp_path: Path,
) -> None:
    """`round(100 * 0.005) == 0`, and Python's banker's rounding makes small
    fractions worse (`round(1 * 0.5) == 0` too), so a naive `round` alone
    would silently give a caller who asked for 0.5% validation data an empty
    validation split. `val_fraction > 0.0` must mean "at least one document
    ends up in val," not "probably some do."
    """
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(100)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=0,
        val_fraction=0.005,
        shard_tokens=1_000_000,
    )

    val_reader = ShardedTokens._from_entries(output_dir, manifest.shards, "val")
    assert len(val_reader) > 0
    assert any(entry.split == "val" for entry in manifest.shards)


def test_val_fraction_outside_zero_one_is_rejected(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    source = tmp_path / "corpus.txt"
    _write_corpus(source, [b"only"])
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="val_fraction"):
        prepare(
            source=source,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            seed=0,
            val_fraction=1.5,
            shard_tokens=10,
        )


# ---------------------------------------------------------------------------
# Other behaviour prepare() is responsible for
# ---------------------------------------------------------------------------


def test_manifest_records_the_run_parameters_and_digests(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [b"hello world", b"goodbye"]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=3,
        val_fraction=0.5,
        shard_tokens=10,
        max_chunk_bytes=999,
    )

    assert manifest.tokenizer_sha256 == sha256_file(tokenizer_path)
    assert manifest.source_sha256 == sha256_file(source)
    assert manifest.seed == 3
    assert manifest.val_fraction == 0.5
    assert manifest.limit is None
    assert manifest.max_chunk_bytes == 999
    assert manifest.nanoscope_version == nanoscope.__version__
    assert 0 < manifest.max_chunk_bytes_observed <= 999
    assert Manifest.load(output_dir / "manifest.json") == manifest


def test_limit_caps_the_number_of_documents_processed(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    docs = [f"doc-{i}".encode() for i in range(10)]
    source = tmp_path / "corpus.txt"
    _write_corpus(source, docs)
    output_dir = tmp_path / "out"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=0,
        val_fraction=0.0,
        shard_tokens=1_000_000,
        limit=3,
    )

    reader = ShardedTokens._from_entries(output_dir, manifest.shards, "train")
    ids = reader.window(0, len(reader)).tolist()
    assert ids.count(END_OF_TEXT_ID) == 3
    # Recorded so the split is reproducible from the manifest alone (design
    # spec section 6): re-deriving the split needs to know how many documents
    # were actually processed, not how many `source` contains.
    assert manifest.limit == 3


def test_output_dir_is_created_if_it_does_not_exist(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    source = tmp_path / "corpus.txt"
    _write_corpus(source, [b"only"])
    output_dir = tmp_path / "does" / "not" / "exist"

    manifest = prepare(
        source=source,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        seed=0,
        val_fraction=0.0,
        shard_tokens=10,
    )
    assert (output_dir / "manifest.json").exists()
    assert manifest.shards
