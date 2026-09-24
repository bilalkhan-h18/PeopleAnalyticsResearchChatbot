"""PDF text extraction and page-aware chunking."""

import re
from dataclasses import dataclass
from pathlib import Path

_REFERENCES_HEADING = re.compile(
    r"^\s*(references|bibliography|reference list|works cited|literature cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    index: int


def read_pdf_pages(path: Path) -> list[Page]:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(Page(number=i, text=text))
    return pages


def clean_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_reference_list(pages: list[Page]) -> list[Page]:
    """Drop the bibliography, which pollutes retrieval with author/journal noise.

    Only a references heading in the last 40% of the document counts, so a
    "References" line in a table of contents doesn't truncate the paper.
    """
    if not pages:
        return pages
    cutoff = int(len(pages) * 0.6)
    for pi in range(len(pages) - 1, cutoff - 1, -1):
        matches = list(_REFERENCES_HEADING.finditer(pages[pi].text))
        if matches:
            head = pages[pi].text[: matches[-1].start()].strip()
            kept = pages[:pi]
            if head:
                kept.append(Page(number=pages[pi].number, text=head))
            return kept
    return pages


def chunk_pages(pages: list[Page], chunk_chars: int, overlap: int) -> list[Chunk]:
    """Split pages into ~chunk_chars chunks that break on paragraph or sentence
    boundaries and remember which pages they came from."""
    # Paragraph-level units tagged with their page number.
    units: list[tuple[str, int]] = []
    for page in pages:
        for para in re.split(r"\n\s*\n", page.text):
            para = " ".join(para.split())
            if not para:
                continue
            if len(para) <= chunk_chars:
                units.append((para, page.number))
            else:
                units.extend((s, page.number) for s in _split_long(para, chunk_chars))

    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    size = 0
    for text, page_no in units:
        if current and size + len(text) + 1 > chunk_chars:
            chunks.append(_make_chunk(current, len(chunks)))
            current = _overlap_tail(current, overlap)
            size = sum(len(t) + 1 for t, _ in current)
        current.append((text, page_no))
        size += len(text) + 1
    if current:
        chunks.append(_make_chunk(current, len(chunks)))
    return chunks


def _split_long(paragraph: str, limit: int) -> list[str]:
    pieces, buf = [], ""
    for sentence in _SENTENCE_END.split(paragraph):
        if buf and len(buf) + len(sentence) + 1 > limit:
            pieces.append(buf)
            buf = ""
        while len(sentence) > limit:  # a "sentence" with no punctuation (tables, etc.)
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def _overlap_tail(units: list[tuple[str, int]], overlap: int) -> list[tuple[str, int]]:
    tail: list[tuple[str, int]] = []
    size = 0
    for text, page_no in reversed(units):
        if size + len(text) > overlap:
            break
        tail.insert(0, (text, page_no))
        size += len(text) + 1
    return tail


def _make_chunk(units: list[tuple[str, int]], index: int) -> Chunk:
    return Chunk(
        text="\n\n".join(t for t, _ in units),
        page_start=units[0][1],
        page_end=units[-1][1],
        index=index,
    )
