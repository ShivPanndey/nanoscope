"""Vocabulary layout, shared by the trainer and the tokenizer.

Its own module because both `train.py` and `tokenizer.py` need these constants
while `train.py` must not import `tokenizer.py`: keeping that dependency
one-way means the trainer's differential test exercises the trainer alone,
with no encoder in the path to muddy a failure.

Layout:

    ids    0..255   raw bytes, id == byte value
    id       256    <|endoftext|>
    ids 257..8191   learned merges, in rank order

The id-equals-byte-value invariant is what makes encoder bugs visible by
inspection rather than only by test failure.
"""

from pydantic import BaseModel, ConfigDict, field_validator

BYTE_TOKENS = 256
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_ID = 256
FIRST_MERGE_ID = 257
DEFAULT_VOCAB_SIZE = 8192


class TokenizerFile(BaseModel):
    """On-disk representation of a trained tokenizer.

    JSON rather than a binary format so a merge table is diffable in review and
    a corrupted one is readable by eye.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    pattern: str
    special_tokens: dict[str, int]
    corpus_sha256: str | None = None
    merges: list[tuple[int, int]]

    @field_validator("merges")
    @classmethod
    def _merges_only_reference_defined_ids(
        cls, merges: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """A merge at rank k may reference only ids below FIRST_MERGE_ID + k,
        excluding the special token id, and no pair may repeat.

        Without this, a truncated or hand-edited file loads into a tokenizer
        whose vocabulary table is subtly wrong -- which shows up as degraded
        compression, not as an error. A duplicate pair is the same defect: it
        validates, encodes to the later rank, and leaves the earlier rank's id
        a permanently dead slot in the vocabulary.
        """
        seen: set[tuple[int, int]] = set()
        for rank, pair in enumerate(merges):
            if pair in seen:
                raise ValueError(f"merge at rank {rank} repeats pair {pair}")
            seen.add(pair)
            limit = FIRST_MERGE_ID + rank
            for token_id in pair:
                if not 0 <= token_id < limit or token_id == END_OF_TEXT_ID:
                    raise ValueError(
                        f"merge at rank {rank} references id {token_id}, "
                        f"which is outside the defined range [0, {limit}) "
                        f"excluding {END_OF_TEXT_ID}"
                    )
        return merges
