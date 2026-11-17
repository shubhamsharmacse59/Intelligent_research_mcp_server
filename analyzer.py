"""
Step 2: the "understanding" part - MAP-REDUCE analysis.

  MAP:    the LLM reads each section separately and writes short notes
  REDUCE: the LLM combines all section notes into one structured report

Why? A full paper is too long to analyse well in one prompt. Small pieces fit
the model's context window and get read carefully.
"""
from pathlib import Path

import config
from llm import ask_llm
from pdf_utils import body_and_references, extract_text, guess_title, split_sections

ANALYST = ("You are an expert research analyst. You read academic papers carefully and "
           "explain them accurately and clearly. Only state what the paper supports.")

REPORT_TEMPLATE = """## Summary
(one paragraph)

## Problem statement
(what problem, and why it matters)

## Proposed method
(the approach explained simply, step by step)

## Datasets and experimental setup

## Key results
(main numbers and comparisons with baselines)

## Main contributions
(3-5 bullet points)

## Strengths

## Limitations and weaknesses
(include ones the authors state and ones you notice)

## Future work

## Key terms explained
(5-8 technical terms from the paper, each in one plain sentence)"""


def _pieces(sections: dict[str, str]) -> list[tuple[str, str]]:
    """Each section becomes one piece; very long sections are cut into parts."""
    pieces = []
    for name, text in sections.items():
        for i in range(0, len(text), config.SECTION_CHARS):
            label = name if len(text) <= config.SECTION_CHARS else f"{name} (part {i // config.SECTION_CHARS + 1})"
            pieces.append((label, text[i:i + config.SECTION_CHARS]))
    return pieces


def analyze(pdf_path: Path, progress=print) -> str:
    """Full analysis of one paper. Returns a Markdown report (also cached to disk)."""
    cache = config.ANALYSES_DIR / f"{pdf_path.stem}.md"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    text = extract_text(pdf_path)
    title = guess_title(pdf_path, text)
    body, refs = body_and_references(split_sections(text))
    pieces = _pieces(body)

    # ---- MAP: notes per section ----
    notes = []
    for n, (label, piece) in enumerate(pieces, start=1):
        progress(f"Reading section {n}/{len(pieces)}: {label}")
        note = ask_llm(
            ANALYST,
            f"This is the '{label}' section of the paper '{title}'.\n\n{piece}\n\n"
            "Write concise notes (max 150 words) on what this section says: key ideas, "
            "methods, datasets, numbers and claims. Keep exact numbers.",
        )
        notes.append(f"### {label}\n{note}")

    # ---- REDUCE: one structured report ----
    progress("Writing the final report")
    report = ask_llm(
        ANALYST,
        f"Below are section-by-section notes on the paper '{title}'.\n\n"
        + "\n\n".join(notes)
        + f"\n\nUsing ONLY these notes, write a detailed analysis in Markdown with exactly "
          f"these headings:\n\n{REPORT_TEMPLATE}",
    )

    final = f"# Analysis: {title}\n\n*File: {pdf_path.name} | Sections read: {len(pieces)}*\n\n{report}"
    cache.write_text(final, encoding="utf-8")
    return final


def compare(analysis_a: str, analysis_b: str) -> str:
    return ask_llm(
        ANALYST,
        f"PAPER A ANALYSIS:\n{analysis_a}\n\nPAPER B ANALYSIS:\n{analysis_b}\n\n"
        "Compare the two papers in Markdown with these headings: Problem, Method, "
        "Datasets, Results, Strengths, Limitations, Which to use when. Use a table "
        "where it helps.",
    )
