"""Shard writing and reading: the on-disk token layout the training loop
reads from at speed.

Design spec section 2: the training loop draws thousands of random
`(batch, seq_len)` windows from a flat, memory-mapped array of ids. Section 7
fixes the on-disk format that makes that cheap: each shard is a headerless
flat array of little-endian `uint16`. Headerless because the manifest already
carries every piece of metadata a header would duplicate, and headerless is
exactly the layout `numpy.memmap` wants. Little-endian is written out
explicitly rather than left to numpy's native byte order, which is
platform-dependent, so a shard set written on one machine still reads
correctly on another.

`ShardWriter` produces that layout; `ShardedTokens` reads it back and serves
a window across a shard seam, so a caller never learns where the boundary
falls (section 8).
"""

import bisect
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import numpy.typing as npt

from nanoscope.data.manifest import ShardEntry, sha256_file

# Headerless little-endian uint16, per design spec section 7. The explicit
# "<" is load-bearing: plain `np.uint16` is numpy's native byte order, which
# is platform-dependent, and this format must be portable across machines.
DTYPE = np.dtype("<u2")

_UINT16_MAX = np.iinfo(np.uint16).max


class ShardWriter:
    """Appends token ids to fixed-size shard files, rolling over to a new
    file every `shard_tokens` ids.

    A shard file is opened lazily, on the first id written to it, so a
    writer that never receives any ids produces no files and `close()`
    returns an empty list. The final shard of a split is typically short;
    that is expected, not an error.
    """

    def __init__(self, output_dir: Path, split: Literal["train", "val"], shard_tokens: int) -> None:
        if shard_tokens <= 0:
            raise ValueError(f"shard_tokens must be positive, got {shard_tokens}")
        self._output_dir = output_dir
        self._split = split
        self._shard_tokens = shard_tokens
        self._shard_index = 0
        self._shard_tokens_written = 0
        self._handle: BinaryIO | None = None
        self._current_path: Path | None = None
        self._entries: list[ShardEntry] = []

    def write(self, ids: Sequence[int]) -> None:
        """Append `ids`, rolling over to a new shard whenever the current one
        fills.

        Every id is checked against `[0, 65535]` before anything is written,
        so a call either writes all of `ids` or none of it. Checked
        explicitly rather than left to numpy's cast, which would wrap an
        out-of-range id silently instead of raising: an id above 65535 would
        poison the corpus in a way no later stage could detect.

        Raises `ValueError` naming the offending id if any id does not fit.
        """
        if not ids:
            return

        ids_array = np.asarray(ids, dtype=np.int64)
        out_of_range = np.flatnonzero((ids_array < 0) | (ids_array > _UINT16_MAX))
        if out_of_range.size:
            offending = int(ids_array[out_of_range[0]])
            raise ValueError(
                f"token id {offending} does not fit in uint16 (must be in [0, {_UINT16_MAX}])"
            )
        payload = ids_array.astype(DTYPE)

        offset = 0
        total = payload.shape[0]
        while offset < total:
            if self._handle is None:
                self._open_shard()
            assert self._handle is not None
            capacity = self._shard_tokens - self._shard_tokens_written
            take = min(capacity, total - offset)
            self._handle.write(payload[offset : offset + take].tobytes())
            self._shard_tokens_written += take
            offset += take
            if self._shard_tokens_written == self._shard_tokens:
                self._close_shard()

    def close(self) -> list[ShardEntry]:
        """Finalize the writer, closing any shard still open, and return the
        `ShardEntry` list for every shard written, in the order written."""
        if self._handle is not None:
            self._close_shard()
        return list(self._entries)

    def _open_shard(self) -> None:
        name = f"{self._split}-{self._shard_index:05d}.bin"
        self._current_path = self._output_dir / name
        self._handle = self._current_path.open("wb")
        self._shard_tokens_written = 0

    def _close_shard(self) -> None:
        assert self._handle is not None
        assert self._current_path is not None
        self._handle.close()
        self._entries.append(
            ShardEntry(
                name=self._current_path.name,
                split=self._split,
                tokens=self._shard_tokens_written,
                sha256=sha256_file(self._current_path),
            )
        )
        self._handle = None
        self._current_path = None
        self._shard_index += 1
        self._shard_tokens_written = 0


class ShardedTokens:
    """Opens a split's shards as `numpy.memmap` objects and presents them as
    one logical sequence of token ids.

    `window(start, length)` returns the ids in `[start, start + length)`,
    concatenating across shard files as needed, so a caller never learns
    where a shard boundary falls. `entries` may hold both splits' shards
    (e.g. a manifest's full `shards` list); only the ones matching `split`
    are opened.
    """

    def __init__(
        self,
        shard_dir: Path,
        entries: Sequence[ShardEntry],
        split: Literal["train", "val"],
    ) -> None:
        # Sorted by name so shard N always precedes shard N+1 regardless of
        # the order `entries` arrived in (e.g. a manifest's shards list,
        # which interleaves both splits in write order).
        split_entries = sorted(
            (entry for entry in entries if entry.split == split), key=lambda e: e.name
        )
        self._shards: list[npt.NDArray[np.uint16]] = [
            np.memmap(shard_dir / entry.name, dtype=DTYPE, mode="r", shape=(entry.tokens,))
            for entry in split_entries
        ]
        self._lengths = [entry.tokens for entry in split_entries]
        self._offsets = [0]
        for shard_length in self._lengths:
            self._offsets.append(self._offsets[-1] + shard_length)
        self._length = self._offsets[-1]

    def __len__(self) -> int:
        return self._length

    def window(self, start: int, length: int) -> npt.NDArray[np.uint16]:
        """Return the ids in `[start, start + length)` as one array.

        Raises `ValueError` if `start` or `length` is negative, or if the
        window runs past the end of the split -- a caller is expected to
        bound `start` so every window it asks for is fully present, the same
        way a fixed `(batch, seq_len)` shape is expected to be filled
        exactly. A window of `length` 0 is valid (including at `start ==
        len(self)`, the position one past the last id) and returns an empty
        array rather than raising: it names a legitimate, if trivial, range.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if length < 0:
            raise ValueError(f"length must be non-negative, got {length}")
        end = start + length
        if end > self._length:
            raise ValueError(
                f"window [{start}, {end}) runs past the end of the split ({self._length} tokens)"
            )
        if length == 0:
            return np.empty(0, dtype=DTYPE)

        shard_index = bisect.bisect_right(self._offsets, start) - 1
        pieces: list[npt.NDArray[np.uint16]] = []
        position = start
        remaining = length
        while remaining > 0:
            shard = self._shards[shard_index]
            shard_start = position - self._offsets[shard_index]
            available = self._lengths[shard_index] - shard_start
            take = min(available, remaining)
            pieces.append(shard[shard_start : shard_start + take])
            position += take
            remaining -= take
            shard_index += 1

        return np.concatenate(pieces) if len(pieces) > 1 else np.array(pieces[0])
