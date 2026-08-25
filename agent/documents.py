"""Markdown document loading with YAML-lite front matter parsing.

The knowledge base uses a small, consistent front-matter subset (plain
``key: value`` lines), so a tiny purpose-built parser avoids pulling in a YAML
dependency while preserving every supplied metadata field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import DocumentMeta

TRUE = {"true", "yes"}
FALSE = {"false", "no"}


@dataclass
class LoadedDocument:
    filename: str
    meta: DocumentMeta
    body: str


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split ``---`` delimited front matter from the Markdown body."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n---", 2)
    # text starts with '---\n'; find the closing delimiter line.
    lines = text.splitlines()
    if lines[0].strip() != "---":
        return {}, text
    fm_lines: list[str] = []
    rest_idx = 1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            rest_idx = i + 1
            break
        fm_lines.append(lines[i])
    body = "\n".join(lines[rest_idx:]).lstrip("\n")
    fm: dict = {}
    for line in fm_lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        low = value.lower()
        if low in TRUE:
            value = True  # type: ignore[assignment]
        elif low in FALSE:
            value = False  # type: ignore[assignment]
        elif value == "":
            value = None  # type: ignore[assignment]
        fm[key] = value
    return fm, body


def meta_from_dict(d: dict) -> DocumentMeta:
    return DocumentMeta(
        document_id=str(d.get("document_id", "")),
        title=str(d.get("title", "")),
        status=str(d.get("status", "unknown")).lower(),
        audience=str(d.get("audience", "unknown")).lower(),
        policy_authority=str(d.get("policy_authority", "none")).lower(),
        effective_date=d.get("effective_date"),
        superseded_date=d.get("superseded_date"),
        last_reviewed=d.get("last_reviewed"),
        supersedes=d.get("supersedes"),
        superseded_by=d.get("superseded_by"),
        customer_answering=(
            d["customer_answering"] if isinstance(d.get("customer_answering"), bool) else None
        ),
    )


def load_documents(kb_dir: Path) -> list[LoadedDocument]:
    docs: list[LoadedDocument] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fm, body = parse_front_matter(text)
        docs.append(
            LoadedDocument(
                filename=path.name,
                meta=meta_from_dict(fm),
                body=body,
            )
        )
    return docs
