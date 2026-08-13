"""The document reader: the `Iterator[bytes]` seam the rest of the pipeline is
built around.

Design spec section 4: `encode` is chunk-local, and a document boundary is
always a chunk boundary, because `\\s*[\\r\\n]+` is one of the split pattern's
alternatives. That makes a newline a safe place to cut a corpus -- encoding a
document in isolation gives exactly the ids it would get inside the whole
corpus. **No other byte offset carries that guarantee**, so this reader never
buffers or seeks past one; it only ever cuts at a `\\n` byte, one line at a
time, and never holds the file in memory.
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
    """
    with path.open("rb") as handle:
        for line in handle:
            yield line[:-1] if line.endswith(b"\n") else line
