"""The tokenizer itself: encoding, decoding, and the str convenience layer."""

from itertools import pairwise
from pathlib import Path

from nanoscope.tokenizer.pretokenize import PATTERN_SOURCE, pretokenize
from nanoscope.tokenizer.train import Pair
from nanoscope.tokenizer.vocab import (
    BYTE_TOKENS,
    END_OF_TEXT,
    END_OF_TEXT_ID,
    FIRST_MERGE_ID,
    TokenizerFile,
)

# Arbitrary binary input has unbounded chunk diversity, so the chunk cache
# needs a ceiling. Clearing wholesale rather than evicting least-recently-used
# keeps this a plain dict: real corpora have few enough unique chunks that the
# limit is never reached, so eviction policy would be untested code.
CACHE_LIMIT = 100_000


def _build_vocab(merges: list[Pair]) -> list[bytes]:
    """Materialise id -> bytes. Merges are in rank order, so this is one pass."""
    vocab = [bytes([i]) for i in range(BYTE_TOKENS)]
    vocab.append(END_OF_TEXT.encode("utf-8"))
    for first, second in merges:
        vocab.append(vocab[first] + vocab[second])
    return vocab


class Tokenizer:
    """Byte-level BPE tokenizer.

    `bytes` is the fundamental domain. `decode(encode(data)) == data` holds for
    every byte string with no exceptions, which is only true because `encode`
    never emits the special token: nothing in the input can map to id 256.

    The `str` methods are a thin wrapper using `surrogatepass`, which is total
    on `str`, so lone surrogates round-trip rather than raising.
    """

    def __init__(self, merges: list[Pair], corpus_sha256: str | None = None) -> None:
        self._merges: list[Pair] = list(merges)
        self._ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(self._merges)}
        self._vocab: list[bytes] = _build_vocab(self._merges)
        self.corpus_sha256 = corpus_sha256
        self._cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, data: bytes) -> list[int]:
        ids: list[int] = []
        for chunk in pretokenize(data):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return list(cached)

        ids = list(chunk)
        while len(ids) >= 2:
            best_rank: int | None = None
            best_index = 0
            for index, pair in enumerate(pairwise(ids)):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_rank is None:
                break
            ids[best_index : best_index + 2] = [FIRST_MERGE_ID + best_rank]

        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        self._cache[chunk] = list(ids)
        return ids

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self._vocab[i] for i in ids)

    def encode_str(self, text: str) -> list[int]:
        return self.encode(text.encode("utf-8", errors="surrogatepass"))

    def decode_str(self, ids: list[int]) -> str:
        return self.decode(ids).decode("utf-8", errors="surrogatepass")

    def save(self, path: Path) -> None:
        document = TokenizerFile(
            pattern=PATTERN_SOURCE,
            special_tokens={END_OF_TEXT: END_OF_TEXT_ID},
            corpus_sha256=self.corpus_sha256,
            merges=self._merges,
        )
        path.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        document = TokenizerFile.model_validate_json(path.read_text(encoding="utf-8"))
        if document.version != 1:
            raise ValueError(
                f"tokenizer file has version {document.version}, but this build "
                "only reads version 1"
            )
        if document.pattern != PATTERN_SOURCE:
            raise ValueError(
                "tokenizer file was trained with a different split pattern; "
                "its merge table is not valid under this one"
            )
        if document.special_tokens != {END_OF_TEXT: END_OF_TEXT_ID}:
            raise ValueError(
                f"tokenizer file declares special tokens {document.special_tokens}, "
                f"but this build hardcodes {{{END_OF_TEXT!r}: {END_OF_TEXT_ID}}}"
            )
        return cls(merges=list(document.merges), corpus_sha256=document.corpus_sha256)
