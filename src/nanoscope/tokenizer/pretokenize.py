"""Pre-tokenization: split bytes into chunks that BPE merges never cross.

The split pattern is cl100k_base's, semantically equivalent to tiktoken's
(MIT licence). Matching tiktoken's split rather than settling for an
approximation is deliberate. The compression comparison promised in DESIGN.md
section 6 is only a clean read on this repo's merge table if both sides split
text the same way; with different splitters it would report the combined
effect of two splitters and two merge tables.
"""

import regex

# cl100k_base's pattern, semantically equivalent to tiktoken's, with the first
# branch enumerated rather than factored: written here as
# `(?i:'s|'t|'re|'ve|'m|'ll|'d)` rather than tiktoken's
# `'(?i:[sdmt]|ll|ve|re)`, which enumerates to the identical 20-string set
# because `'` has no case mapping. Current tiktoken additionally uses
# possessive quantifiers as a ReDoS hardening, not reproduced here. Needs
# \p{L} and \p{N}, which the standard library `re` does not support -- hence
# the `regex` dependency.
PATTERN_SOURCE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

_PATTERN = regex.compile(PATTERN_SOURCE)


def pretokenize(data: bytes) -> list[bytes]:
    """Split `data` into pre-token chunks that rejoin to exactly `data`.

    The pattern is defined over `str` while the tokenizer's domain is `bytes`.
    `surrogateescape` bridges the two: it is total over all byte strings and
    byte-exact on re-encode, so no input is rejected and none is altered.

    `finditer` rather than `findall`: `_PATTERN` has zero capturing groups
    (`(?i:...)` is a non-capturing inline-flag group), so `findall` here would
    return the same whole-match strings `finditer` does -- `finditer` is used
    because each match needs `.encode(..., errors="surrogateescape")` applied
    via `.group()`, not because of a group-content trap.
    """
    text = data.decode("utf-8", errors="surrogateescape")
    return [m.group().encode("utf-8", errors="surrogateescape") for m in _PATTERN.finditer(text)]
