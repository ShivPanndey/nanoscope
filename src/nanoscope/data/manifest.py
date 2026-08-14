"""The manifest: the record that makes a shard set reproducible.

A manifest pins a set of token shards to the exact tokenizer and source file that
produced them, and to the seed and fraction that produced their train/validation
split. Training on shards produced by a different tokenizer is a silent-garbage
failure -- the ids still decode, just into the wrong text -- so the manifest exists
to make that pairing checkable rather than assumed.

`version` exists for the same reason `TokenizerFile.version` does: it is worthless
unless something rejects a version it does not understand. `Manifest.load` gates on
it and names both versions in the error, following `Tokenizer.load`'s precedent.
"""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

CURRENT_VERSION = 1

# Large enough that hashing a multi-gigabyte shard costs a handful of syscalls,
# small enough that the working set stays trivial regardless of shard size.
_HASH_CHUNK_BYTES = 64 * 1024


class ShardEntry(BaseModel):
    """One shard file on disk: enough to name it, place it in a split, and
    verify its contents without re-tokenizing anything."""

    model_config = ConfigDict(extra="forbid")

    name: str
    split: Literal["train", "val"]
    tokens: int
    sha256: str


class Manifest(BaseModel):
    """On-disk record of how a shard set was produced.

    JSON rather than a binary format, for the same reason as `TokenizerFile`: a
    manifest is small, and being diffable in review and readable by eye matters
    more than compactness.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = CURRENT_VERSION
    tokenizer_sha256: str
    source_sha256: str
    seed: int
    val_fraction: float
    # `None` means "no limit was applied": every document `source` contains was
    # processed. Recording this is what makes design spec section 6's claim --
    # "the split is reproducible from the manifest alone" -- actually true for a
    # `--limit`ed run: the split depends on the document count actually
    # processed, not on `source`'s full document count, and `source_sha256`
    # alone cannot distinguish the two. Optional so a manifest predating this
    # field still loads.
    limit: int | None = None
    # The configured ceiling `prepare` was run with, and the running maximum it
    # actually observed. Recording the configured value, not just what was
    # observed, matters for the same reason as `limit`: without it, a reader
    # cannot tell whether `max_chunk_bytes_observed` reflects "nothing came
    # close to the ceiling" or "the ceiling itself was unusually high or low"
    # -- nor, combined with `limit`, whether it was measured over the whole
    # corpus or only a processed subset of it. Optional for the same
    # backward-compatibility reason as `limit`.
    max_chunk_bytes: int | None = None
    max_chunk_bytes_observed: int
    nanoscope_version: str
    shards: list[ShardEntry]

    def save(self, path: Path) -> None:
        """Write this manifest to `path` as JSON.

        Raises whatever `path.write_text` raises (e.g. `OSError` if the
        destination is not writable); the document itself is always valid,
        since it is built from this manifest's own validated state.
        """
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Read and validate a manifest written by `save`.

        Raises `pydantic.ValidationError` (a `ValueError` subclass) for an
        unknown or malformed field, and plain `ValueError` for a version this
        build does not know how to read.
        """
        manifest = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.version != CURRENT_VERSION:
            raise ValueError(
                f"manifest has version {manifest.version}, but this build only "
                f"reads version {CURRENT_VERSION}"
            )
        return manifest


def sha256_file(path: Path) -> str:
    """Digest a file's contents without loading it into memory at once.

    Shard files can be large, so this reads in fixed-size chunks rather than
    calling `path.read_bytes()`.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
