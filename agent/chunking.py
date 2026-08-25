"""Markdown-aware chunking.

Each ``##`` section becomes one chunk; the H1 document title is kept as the
root of the heading path so every chunk retains its policy context, and the
preamble before the first heading is anchored under the document title.
Sections in this corpus are 150–700 characters, small enough to stay atomic:
merging fragments into neighbours corrupted citation headings (bug diary #1).
"""

from __future__ import annotations

import re

from .contracts import Chunk
from .documents import LoadedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_sections(body: str) -> list[tuple[tuple[str, ...], str]]:
    """Return [(heading_path, section_text)] for a Markdown body."""
    h1_title: str | None = None
    h2_title: str | None = None
    current_path: tuple[str, ...] | None = None
    buf: list[str] = []
    sections: list[tuple[tuple[str, ...], str]] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        if current_path is None:
            sections.append((("__intro__",), text))
        else:
            sections.append((current_path, text))

    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                h1_title = title
                h2_title = None
                current_path = None  # H1 itself opens no chunk
                continue
            if level == 2 or h2_title is None:
                h2_title = title
                base = (h1_title,) if h1_title else ()
                current_path = base + (title,)
            else:  # H3+ nested under its H2
                base = (h1_title,) if h1_title else ()
                current_path = base + (f"{h2_title} > {title}",)
        buf.append(line)
    flush()
    return sections


def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    raw = split_sections(doc.body)

    chunks: list[Chunk] = []
    for i, (path, text) in enumerate(raw):
        if path == ("__intro__",):
            heading = (doc.meta.title,)
        else:
            heading = path or (doc.meta.title,)
        display_heading = " > ".join(heading)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.filename}#{i}:{display_heading}",
                filename=doc.filename,
                heading_path=heading,
                text=text,
                meta=doc.meta,
            )
        )
    return chunks


def chunk_all(docs: list[LoadedDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d))
    return chunks
