"""Tests for the manifest model: the record that pairs a shard set with the
exact tokenizer and source file that produced it."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import nanoscope
from nanoscope.data.manifest import CURRENT_VERSION, Manifest, ShardEntry, sha256_file


def _valid_manifest() -> Manifest:
    return Manifest(
        tokenizer_sha256="a" * 64,
        source_sha256="b" * 64,
        seed=7,
        val_fraction=0.1,
        limit=1000,
        max_chunk_bytes=1024,
        max_chunk_bytes_observed=54,
        nanoscope_version=nanoscope.__version__,
        shards=[
            ShardEntry(name="train-00000.bin", split="train", tokens=10, sha256="c" * 64),
            ShardEntry(name="val-00000.bin", split="val", tokens=3, sha256="d" * 64),
        ],
    )


def _valid_manifest_json() -> dict[str, object]:
    """A plain dict that `json.dumps` can serialize, matching `_valid_manifest`
    field for field, for tests that need to corrupt one field at a time."""
    return {
        "version": CURRENT_VERSION,
        "tokenizer_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "seed": 7,
        "val_fraction": 0.1,
        "limit": 1000,
        "max_chunk_bytes": 1024,
        "max_chunk_bytes_observed": 54,
        "nanoscope_version": nanoscope.__version__,
        "shards": [
            {"name": "train-00000.bin", "split": "train", "tokens": 10, "sha256": "c" * 64},
        ],
    }


def test_a_saved_manifest_round_trips_through_json(tmp_path: Path) -> None:
    original = _valid_manifest()
    path = tmp_path / "manifest.json"
    original.save(path)
    assert Manifest.load(path) == original


def test_a_manifest_written_before_limit_and_max_chunk_bytes_existed_still_loads(
    tmp_path: Path,
) -> None:
    """`limit` and `max_chunk_bytes` are optional fields added after this
    manifest shape was first shipped. A file that predates them (neither key
    present at all, not even `null`) must still load, defaulting both to
    `None` rather than being rejected as unrecognized or malformed.
    """
    fields = _valid_manifest_json()
    del fields["limit"]
    del fields["max_chunk_bytes"]
    path = tmp_path / "pre-existing-fields.json"
    path.write_text(json.dumps(fields), encoding="utf-8")

    manifest = Manifest.load(path)
    assert manifest.limit is None
    assert manifest.max_chunk_bytes is None


def test_load_rejects_a_file_with_an_unknown_field(tmp_path: Path) -> None:
    """A misspelled field (e.g. tokenizer_sha_256) must not be silently dropped.

    With extra fields forbidden, a typo'd key fails loudly instead of loading
    as if the field it was meant to be were simply absent, which would leave
    provenance silently missing.
    """
    fields = _valid_manifest_json()
    del fields["tokenizer_sha256"]
    fields["tokenizer_sha_256"] = "a" * 64
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        Manifest.load(path)


def test_shard_entry_also_rejects_an_unknown_field(tmp_path: Path) -> None:
    fields = _valid_manifest_json()
    shard = fields["shards"]
    assert isinstance(shard, list)
    shard[0]["digest"] = shard[0].pop("sha256")
    path = tmp_path / "typo-shard.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        Manifest.load(path)


def test_shard_entry_rejects_a_split_that_is_neither_train_nor_val() -> None:
    with pytest.raises(ValidationError):
        ShardEntry(name="oops-00000.bin", split="test", tokens=1, sha256="a" * 64)  # type: ignore[arg-type]


def test_load_rejects_a_file_with_an_unsupported_version(tmp_path: Path) -> None:
    """A version this build does not know how to read must fail, not load
    partially, and the error must name both the version found and the one
    expected."""
    fields = _valid_manifest_json()
    fields["version"] = CURRENT_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    with pytest.raises(ValueError, match="version") as exc_info:
        Manifest.load(path)
    message = str(exc_info.value)
    assert str(CURRENT_VERSION + 1) in message
    assert str(CURRENT_VERSION) in message


def test_sha256_file_matches_a_digest_computed_independently(tmp_path: Path) -> None:
    """Computed with a fresh `hashlib.sha256` call on the same bytes, not by
    calling `sha256_file` twice, so the test can actually catch a wrong digest."""
    data = os.urandom(200_000)  # spans several read chunks
    path = tmp_path / "shard.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == expected


def test_sha256_file_of_an_empty_file_matches_the_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_tokens_in_sums_only_the_named_splits_shards() -> None:
    """The sum lives on `Manifest` rather than at the call site so a caller
    can report a split's size without holding manifest-internal knowledge of
    how shards are laid out. Asserted per split and with unequal totals, so a
    filter that matched the wrong split, or none, cannot pass.
    """
    manifest = Manifest(
        tokenizer_sha256="a" * 64,
        source_sha256="b" * 64,
        seed=0,
        val_fraction=0.25,
        max_chunk_bytes_observed=4,
        nanoscope_version=nanoscope.__version__,
        shards=[
            ShardEntry(name="train-00000.bin", split="train", tokens=10, sha256="c" * 64),
            ShardEntry(name="val-00000.bin", split="val", tokens=3, sha256="d" * 64),
            ShardEntry(name="train-00001.bin", split="train", tokens=7, sha256="e" * 64),
        ],
    )

    assert manifest.tokens_in("train") == 17
    assert manifest.tokens_in("val") == 3


def test_tokens_in_is_zero_for_a_split_with_no_shards() -> None:
    manifest = Manifest(
        tokenizer_sha256="a" * 64,
        source_sha256="b" * 64,
        seed=0,
        val_fraction=0.0,
        max_chunk_bytes_observed=4,
        nanoscope_version=nanoscope.__version__,
        shards=[ShardEntry(name="train-00000.bin", split="train", tokens=10, sha256="c" * 64)],
    )

    assert manifest.tokens_in("val") == 0
