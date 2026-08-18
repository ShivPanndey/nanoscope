"""The document reader: the `Iterator[bytes]` seam the rest of the pipeline is
built around.

Design spec section 4: a document boundary is **not always** a chunk boundary
-- `pretokenize` sometimes absorbs a trailing newline into the chunk before
it rather than isolating it as its own chunk. What actually makes `\\n` a
safe place to cut a corpus is narrower: it is the document delimiter, and
this reader hands each document to the tokenizer on its own, never joined to
a neighbour's bytes, so no chunk downstream of this reader can ever span
from one document's content into the next. **No other byte offset carries
that guarantee**, so this reader never buffers or seeks past one; it only
ever cuts at a `\\n` byte, one line at a time, and never holds the file in
memory.
"""

from collections.abc import Iterator
from pathlib import Path


def iter_documents(path: Path) -> Iterator[bytes]:
    """Yield each document in `path`, one at a time, without holding the file
    in memory.

    The corpus is one document per line: `path` is opened in binary mode and
    read line by line, and each line's trailing `\\n` is stripped before it is
    yielded. An empty file yields nothing. A file whose last line has no
    trailing newline still yields that line. A blank line yields `b""` rather
    than being skipped, since an empty line is still a document, just an empty
    one.

    **`\\n` is the only line terminator recognised, deliberately.** A CRLF
    corpus therefore yields documents that each end in a literal `\\r`, which
    the tokenizer will turn into a real token at the end of every document.
    Stripping `\\r` here would be the wrong trade: it would also strip a `\\r`
    that is genuinely the last byte of a document in an LF corpus, and this
    reader's law is that `iter_documents` recovers exactly the documents that
    were written (the round-trip property test in
    `tests/test_documents.py`). Byte-exactness is what makes the manifest's
    `source_sha256` mean something, so line-ending conversion belongs to
    whoever prepares the corpus, not here. TinyStories, the corpus this
    pipeline targets, is LF.
    """
    with path.open("rb") as handle:
        for line in handle:
            yield line[:-1] if line.endswith(b"\n") else line
