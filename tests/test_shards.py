"""Tests for shard writing and reading: the on-disk layout the training loop
reads from and the writer that produces it.

Design spec section 7: shards are headerless flat arrays of little-endian
`uint16`, since the manifest already carries the metadata a header would
duplicate. Section 8: `ShardedTokens` serves a window across a shard seam so
that a caller never learns where the boundary falls -- the straddling case is
the one a naive implementation gets wrong, so it is tested directly below
(known seams) and then over random bounds (property test).
"""

import struct
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import nanoscope
from nanoscope.data.manifest import Manifest, ShardEntry, sha256_file
from nanoscope.data.shards import DTYPE, ShardedTokens, ShardWriter
from nanoscope.tokenizer import Tokenizer

# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------


def test_writing_fewer_ids_than_shard_tokens_produces_one_short_shard(
    tmp_path: Path,
) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    writer.write([1, 2, 3])
    entries = writer.close()
    assert [e.name for e in entries] == ["train-00000.bin"]
    assert entries[0].tokens == 3
    assert entries[0].split == "train"
    on_disk = np.fromfile(tmp_path / "train-00000.bin", dtype="<u2")
    assert on_disk.tolist() == [1, 2, 3]


def test_writing_exactly_shard_tokens_ids_produces_one_full_shard_and_no_empty_next(
    tmp_path: Path,
) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=4)
    writer.write([1, 2, 3, 4])
    entries = writer.close()
    assert len(entries) == 1
    assert entries[0].tokens == 4


def test_writing_more_ids_than_shard_tokens_rolls_over_to_a_new_shard(
    tmp_path: Path,
) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=4)
    writer.write(list(range(10)))
    entries = writer.close()
    assert [e.name for e in entries] == [
        "train-00000.bin",
        "train-00001.bin",
        "train-00002.bin",
    ]
    assert [e.tokens for e in entries] == [4, 4, 2]
    assert sum(e.tokens for e in entries) == 10


def test_rollover_is_correct_even_when_a_single_write_call_spans_two_shards(
    tmp_path: Path,
) -> None:
    """A write() call whose ids don't align to the shard boundary must still
    split correctly mid-call, not just at call boundaries."""
    writer = ShardWriter(tmp_path, "train", shard_tokens=4)
    writer.write([0, 1, 2])  # 3 into shard 0
    writer.write([3, 4, 5, 6, 7])  # fills shard 0, then all of shard 1
    entries = writer.close()
    assert [e.tokens for e in entries] == [4, 4]
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    np.testing.assert_array_equal(reader.window(0, 8), np.arange(8, dtype=np.uint16))


def test_an_empty_write_call_is_a_no_op(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=4)
    writer.write([])
    entries = writer.close()
    assert entries == []
    assert list(tmp_path.iterdir()) == []


def test_val_split_shards_are_named_and_tagged_for_val(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "val", shard_tokens=4)
    writer.write([1, 2])
    entries = writer.close()
    assert entries[0].name == "val-00000.bin"
    assert entries[0].split == "val"


def test_each_shard_entrys_digest_matches_the_files_actual_sha256(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=4)
    writer.write(list(range(10)))
    entries = writer.close()
    for entry in entries:
        assert entry.sha256 == sha256_file(tmp_path / entry.name)


def test_a_shard_tokens_of_zero_or_negative_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shard_tokens"):
        ShardWriter(tmp_path, "train", shard_tokens=0)
    with pytest.raises(ValueError, match="shard_tokens"):
        ShardWriter(tmp_path, "train", shard_tokens=-1)


def test_an_id_above_uint16_max_is_rejected_and_named_in_the_error(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    with pytest.raises(ValueError, match="65536"):
        writer.write([1, 2, 65536])


def test_a_negative_id_is_also_rejected(tmp_path: Path) -> None:
    """A negative id does not "fit" in uint16 either -- casting would wrap it
    to a huge unsigned value rather than raising, so it must be caught here."""
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    with pytest.raises(ValueError, match="-1"):
        writer.write([1, -1])


def test_the_uint16_boundary_value_65535_is_accepted(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    writer.write([65535])
    entries = writer.close()
    assert entries[0].tokens == 1
    on_disk = np.fromfile(tmp_path / entries[0].name, dtype="<u2")
    assert on_disk.tolist() == [65535]


def test_a_rejected_write_call_writes_nothing_from_that_call(tmp_path: Path) -> None:
    """Validation happens before any bytes from the call are written, so a
    bad id anywhere in the call does not leave a partial write on disk."""
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    writer.write([1, 2, 3])
    with pytest.raises(ValueError, match="70000"):
        writer.write([4, 5, 70000])
    entries = writer.close()
    assert entries[0].tokens == 3


def test_shards_are_written_little_endian_regardless_of_host_byte_order(
    tmp_path: Path,
) -> None:
    """258 (0x0102) has distinct bytes when read big-endian (0x0201 = 513), so
    this fails if the writer ever used native byte order on a platform where
    that differs from little-endian."""
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    writer.write([258])
    entries = writer.close()
    raw = (tmp_path / entries[0].name).read_bytes()
    assert struct.unpack("<H", raw)[0] == 258
    assert struct.unpack(">H", raw)[0] == 513


# ---------------------------------------------------------------------------
# ShardedTokens
# ---------------------------------------------------------------------------


def _write_ramp(
    tmp_path: Path, split: Literal["train", "val"], count: int, shard_tokens: int
) -> list[ShardEntry]:
    writer = ShardWriter(tmp_path, split, shard_tokens)
    writer.write(list(range(count)))
    return writer.close()


def test_a_window_within_a_single_shard_matches_the_unsharded_array(tmp_path: Path) -> None:
    entries = _write_ramp(tmp_path, "train", count=20, shard_tokens=4)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    reference = np.arange(20, dtype=np.uint16)
    np.testing.assert_array_equal(reader.window(1, 2), reference[1:3])


def test_window_straddling_each_seam_matches_the_unsharded_array(tmp_path: Path) -> None:
    """A ramp across five shards. For every shard seam, several windows that
    start before the seam and end after it must equal the same slice taken
    from the flat unsharded array -- the case a naive implementation (e.g.
    one that only reads from the shard containing `start`) gets wrong.
    """
    shard_tokens = 4
    count = 20
    entries = _write_ramp(tmp_path, "train", count=count, shard_tokens=shard_tokens)
    assert len(entries) == 5  # 20 / 4, confirms the ramp spans >= 3 shards
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    reference = np.arange(count, dtype=np.uint16)

    seams = [shard_tokens * i for i in range(1, len(entries))]  # [4, 8, 12, 16]
    for seam in seams:
        for start, length in [(seam - 1, 2), (seam - 2, 4), (seam - 3, 6)]:
            if start < 0 or start + length > count:
                continue
            assert start < seam < start + length  # sanity: really does straddle
            got = reader.window(start, length)
            np.testing.assert_array_equal(got, reference[start : start + length])


def test_a_window_spanning_an_entire_shard_matches_the_unsharded_array(
    tmp_path: Path,
) -> None:
    shard_tokens = 4
    entries = _write_ramp(tmp_path, "train", count=20, shard_tokens=shard_tokens)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    reference = np.arange(20, dtype=np.uint16)
    # Starts one before shard 1 (offset 4) and ends one past its end (offset 8):
    # covers all of shard 1 plus one token from each neighbour.
    np.testing.assert_array_equal(reader.window(3, 6), reference[3:9])


def test_window_over_random_bounds_matches_the_unsharded_array() -> None:
    """Property test: random (start, length) pairs, not only at the seams, so
    a window spanning an entire shard is exercised too.

    The shard files are written once, before `@given` runs, and the nested
    `check` function takes no pytest fixture -- only Hypothesis-drawn values
    -- so there is no function-scoped fixture reused across examples for the
    health check to flag.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        shard_dir = Path(tmp_dir)
        count = 37
        entries = _write_ramp(shard_dir, "train", count=count, shard_tokens=5)
        reader = ShardedTokens._from_entries(shard_dir, entries, "train")
        reference = np.arange(count, dtype=np.uint16)

        @given(data=st.data())
        def check(data: st.DataObject) -> None:
            start = data.draw(st.integers(min_value=0, max_value=count))
            length = data.draw(st.integers(min_value=0, max_value=count - start))
            got = reader.window(start, length)
            np.testing.assert_array_equal(got, reference[start : start + length])

        check()


def test_a_zero_length_window_returns_an_empty_array(tmp_path: Path) -> None:
    entries = _write_ramp(tmp_path, "train", count=10, shard_tokens=4)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    result = reader.window(3, 0)
    assert result.shape == (0,)
    assert result.dtype == np.uint16


def test_a_zero_length_window_at_the_very_end_is_valid(tmp_path: Path) -> None:
    entries = _write_ramp(tmp_path, "train", count=10, shard_tokens=4)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    result = reader.window(10, 0)
    assert result.shape == (0,)


def test_a_window_running_past_the_end_raises(tmp_path: Path) -> None:
    entries = _write_ramp(tmp_path, "train", count=10, shard_tokens=4)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    with pytest.raises(ValueError, match="10"):
        reader.window(8, 3)


def test_a_negative_start_raises(tmp_path: Path) -> None:
    entries = _write_ramp(tmp_path, "train", count=10, shard_tokens=4)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    with pytest.raises(ValueError, match="start"):
        reader.window(-1, 1)


def test_a_negative_length_raises(tmp_path: Path) -> None:
    entries = _write_ramp(tmp_path, "train", count=10, shard_tokens=4)
    reader = ShardedTokens._from_entries(tmp_path, entries, "train")
    with pytest.raises(ValueError, match="length"):
        reader.window(0, -1)


def test_shardedtokens_only_reads_shards_belonging_to_the_requested_split(
    tmp_path: Path,
) -> None:
    train_writer = ShardWriter(tmp_path, "train", shard_tokens=4)
    train_writer.write(list(range(10)))
    train_entries = train_writer.close()

    val_writer = ShardWriter(tmp_path, "val", shard_tokens=4)
    val_writer.write(list(range(100, 106)))
    val_entries = val_writer.close()

    all_entries = train_entries + val_entries
    train_reader = ShardedTokens._from_entries(tmp_path, all_entries, "train")
    val_reader = ShardedTokens._from_entries(tmp_path, all_entries, "val")

    assert len(train_reader) == 10
    assert len(val_reader) == 6
    np.testing.assert_array_equal(train_reader.window(0, 10), np.arange(10, dtype=np.uint16))
    np.testing.assert_array_equal(val_reader.window(0, 6), np.arange(100, 106, dtype=np.uint16))


def test_a_split_with_no_shards_has_length_zero(tmp_path: Path) -> None:
    reader = ShardedTokens._from_entries(tmp_path, [], "train")
    assert len(reader) == 0
    assert reader.window(0, 0).shape == (0,)


# ---------------------------------------------------------------------------
# ShardedTokens.open -- design spec section 8's tokenizer-digest check
# ---------------------------------------------------------------------------


def _write_manifest_and_shards(tmp_path: Path, tokenizer_path: Path) -> Path:
    """A minimal end-to-end fixture: a tokenizer file, one train shard, and a
    manifest naming the tokenizer's real sha256. Returns the manifest path."""
    Tokenizer([]).save(tokenizer_path)
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    writer.write([1, 2, 3])
    entries = writer.close()
    manifest = Manifest(
        tokenizer_sha256=sha256_file(tokenizer_path),
        source_sha256="a" * 64,
        seed=0,
        val_fraction=0.0,
        max_chunk_bytes_observed=1,
        nanoscope_version=nanoscope.__version__,
        shards=entries,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def test_open_with_the_matching_tokenizer_reads_the_same_data_as_from_entries(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = _write_manifest_and_shards(tmp_path, tokenizer_path)
    manifest = Manifest.load(manifest_path)

    opened = ShardedTokens.open(manifest_path, tokenizer_path, "train")
    raw = ShardedTokens._from_entries(tmp_path, manifest.shards, "train")
    np.testing.assert_array_equal(opened.window(0, 3), raw.window(0, 3))


def test_open_rejects_a_tokenizer_that_does_not_match_the_manifests_digest(
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = _write_manifest_and_shards(tmp_path, tokenizer_path)

    # A different tokenizer file: same class, different (empty vs. one that
    # carries a corpus digest) content, so a different sha256 on disk.
    other_tokenizer_path = tmp_path / "other-tokenizer.json"
    Tokenizer([], corpus_sha256="c" * 64).save(other_tokenizer_path)

    with pytest.raises(ValueError, match="tokenizer digest mismatch"):
        ShardedTokens.open(manifest_path, other_tokenizer_path, "train")


def test_the_raw_constructor_is_not_the_public_entry_point(tmp_path: Path) -> None:
    """The old `(shard_dir, entries, split)` signature is gone from the public
    constructor entirely, not just discouraged: `ShardedTokens()` takes no
    arguments, so calling it the old way raises `TypeError` before any shard
    file is ever touched. The only public way to get a populated instance is
    `open()`, which performs the digest check."""
    with pytest.raises(TypeError):
        ShardedTokens(tmp_path, [], "train")  # type: ignore[call-arg]

    assert len(ShardedTokens()) == 0


# ---------------------------------------------------------------------------
# Shard ordering and content verification
# ---------------------------------------------------------------------------


def _handmade_entry(shard_dir: Path, name: str, ids: list[int]) -> ShardEntry:
    """Write a shard file under an arbitrary name and describe it.

    Names are chosen by the caller here rather than by `ShardWriter`, which
    is the point: the index-parsing order has to be exercised at indices no
    test could reach by actually writing 100,000 shards.
    """
    path = shard_dir / name
    path.write_bytes(np.asarray(ids, dtype=DTYPE).tobytes())
    return ShardEntry(
        name=name,
        split="train",
        tokens=len(ids),
        sha256=sha256_file(path),
    )


def test_shards_are_ordered_by_index_not_lexicographically(tmp_path: Path) -> None:
    """Shard names are zero-padded to five digits, so a lexicographic sort
    silently reorders the token stream the moment the padding overflows:
    `train-100000.bin` sorts before `train-99999.bin`. Reaching that by
    writing 100,000 real shards is not testable, so the entries are built by
    hand at exactly the boundary.

    The failure this pins is silent -- no exception, just training data in
    the wrong order -- which is why it is worth a test despite needing a
    trillion tokens at the default shard size to occur naturally.
    """
    entries = [
        _handmade_entry(tmp_path, "train-99999.bin", [1, 2]),
        _handmade_entry(tmp_path, "train-100000.bin", [3, 4]),
    ]

    for order in (entries, list(reversed(entries))):
        reader = ShardedTokens._from_entries(tmp_path, order, "train")
        np.testing.assert_array_equal(reader.window(0, 4), np.array([1, 2, 3, 4], dtype=DTYPE))


def test_a_shard_name_the_writer_could_not_have_produced_is_rejected(
    tmp_path: Path,
) -> None:
    """An unrecognised name means the manifest and the writer disagree about
    the on-disk layout. Falling back to some arbitrary order would hide that
    behind data served in an order nobody chose."""
    entries = [_handmade_entry(tmp_path, "train-shard-one.bin", [1, 2])]

    with pytest.raises(ValueError, match="is not of the form"):
        ShardedTokens._from_entries(tmp_path, entries, "train")


def test_open_can_verify_shard_digests_and_rejects_a_corrupted_shard(
    tmp_path: Path,
) -> None:
    """`ShardEntry.sha256` is recorded so shard contents can be "verified
    without re-tokenizing anything" (the manifest module's own docstring),
    but nothing could act on that promise until `verify_shards`.

    The corruption here keeps the file length identical, which is exactly the
    case `np.memmap` cannot catch: it raises only when a file is *shorter*
    than its recorded token count, so a same-length corruption maps cleanly
    and reads garbage. `prepare` leaves already-written shards behind when a
    run fails partway, so a partially overwritten shard set is reachable.
    """
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = _write_manifest_and_shards(tmp_path, tokenizer_path)

    # Same length, different contents.
    shard_path = tmp_path / "train-00000.bin"
    original = shard_path.read_bytes()
    shard_path.write_bytes(np.asarray([9, 9, 9], dtype=DTYPE).tobytes())
    assert len(shard_path.read_bytes()) == len(original)

    # The default path still opens it: verification is opt-in because it
    # re-reads the whole split, which the training loop must not pay for on
    # every open.
    assert len(ShardedTokens.open(manifest_path, tokenizer_path, "train")) == 3

    with pytest.raises(ValueError, match="shard digest mismatch"):
        ShardedTokens.open(manifest_path, tokenizer_path, "train", verify_shards=True)


def test_verify_shards_accepts_an_intact_shard_set(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    manifest_path = _write_manifest_and_shards(tmp_path, tokenizer_path)

    reader = ShardedTokens.open(manifest_path, tokenizer_path, "train", verify_shards=True)
    np.testing.assert_array_equal(reader.window(0, 3), np.array([1, 2, 3], dtype=DTYPE))
