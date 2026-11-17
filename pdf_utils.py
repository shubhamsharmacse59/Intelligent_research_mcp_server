"""
Step 1: read the PDF and organise its text.

PDF -> raw text -> cleaned text -> sections (abstract, introduction, method, ...)
                                -> references (kept separately)
"""
import re
from pathlib import Path

import pymupdf

HEADING = re.compile(
    r"^\s*(?:[IVX]+\.|\d+(?:\.\d+)*\.?)?\s*"
    r"(abstract|introduction|related work|background|preliminaries|method(?:s|ology)?|"
    r"approach|proposed method|model|architecture|experiments?|experimental setup|"
    r"evaluation|results?|results and discussion|discussion|analysis|ablation stud(?:y|ies)|"
    r"conclusions?|limitations|future work|references|bibliography|appendix|acknowledg(?:e)?ments?)\s*$",
    re.IGNORECASE,
)


def extract_text(pdf_path: Path) -> str:
    with pymupdf.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                     # re-join hyphenated words
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)   # drop page-number lines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def guess_title(pdf_path: Path, text: str) -> str:
    """PDF metadata title if present, otherwise the first meaningful line."""
    with pymupdf.open(pdf_path) as doc:
        meta_title = (doc.metadata or {}).get("title", "").strip()
    if len(meta_title) > 10:
        return meta_title
    for line in text.split("\n")[:15]:
        if len(line.strip()) > 15:
            return line.strip()
    return pdf_path.stem


def split_sections(text: str) -> dict[str, str]:
    """Split on known headings. Text before the first heading = 'front matter'."""
    sections, name, lines = {}, "front matter", []
    for line in text.split("\n"):
        m = HEADING.match(line)
        if m and len(line) < 60:
            if lines:
                sections[name] = sections.get(name, "") + "\n".join(lines)
            name, lines = m.group(1).lower(), []
        else:
            lines.append(line)
    if lines:
        sections[name] = sections.get(name, "") + "\n".join(lines)
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def body_and_references(sections: dict[str, str]) -> tuple[dict[str, str], str]:
    refs = sections.get("references", "") or sections.get("bibliography", "")
    body = {k: v for k, v in sections.items()
            if k not in ("references", "bibliography", "acknowledgments", "acknowledgements",
                         "acknowledgment", "acknowledgement")}
    return body, refs


def split_references(refs_text: str) -> list[str]:
    parts = re.split(r"\s*\[\d+\]\s*", refs_text) if re.search(r"\[\d+\]", refs_text) \
        else re.split(r"\n\s*\n", refs_text)
    return [" ".join(p.split()) for p in parts if len(p.strip()) > 30]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]
