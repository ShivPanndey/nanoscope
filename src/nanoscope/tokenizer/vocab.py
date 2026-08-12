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

BYTE_TOKENS = 256
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_ID = 256
FIRST_MERGE_ID = 257
DEFAULT_VOCAB_SIZE = 8192
