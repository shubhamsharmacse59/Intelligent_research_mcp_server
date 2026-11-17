"""
MCP SERVER - exposes the paper-analysis abilities as tools.

Any MCP client can use these tools. Our client is the Streamlit app (app.py),
which starts this file and talks to it over stdio.
"""
import re

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import config
import analyzer
import rag
from pdf_utils import body_and_references, extract_text, split_references, split_sections

mcp = MCPServer("research-paper-analyst")


def friendly_errors(fn):
    """Send a clear error message to the client instead of a generic failure."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


def _pdf(file_name: str):
    path = config.PAPERS_DIR / file_name
    if not path.exists():
        raise ValueError(f"'{file_name}' not found in the papers folder.")
    return path


@mcp.tool()
@friendly_errors
def list_papers() -> list[dict]:
    """List the PDFs in the papers folder and whether each has been analysed."""
    return [{"file": p.name, "analysed": (config.ANALYSES_DIR / f"{p.stem}.md").exists()}
            for p in sorted(config.PAPERS_DIR.glob("*.pdf"))]


@mcp.tool()
@friendly_errors
def analyze_paper(file_name: str) -> str:
    """Read a research paper PDF and return a complete structured analysis:
    summary, problem, method, datasets, results, contributions, strengths,
    limitations, future work and key terms. Cached after the first run."""
    return analyzer.analyze(_pdf(file_name), progress=lambda msg: None)


@mcp.tool()
@friendly_errors
def ask_paper(file_name: str, question: str) -> dict:
    """Answer a question about one paper using the most relevant passages (RAG)."""
    return rag.answer(_pdf(file_name), question)


@mcp.tool()
@friendly_errors
def compare_papers(file_a: str, file_b: str) -> str:
    """Compare two papers: problem, method, datasets, results, strengths, limitations."""
    a = analyzer.analyze(_pdf(file_a), progress=lambda msg: None)
    b = analyzer.analyze(_pdf(file_b), progress=lambda msg: None)
    return analyzer.compare(a, b)


@mcp.tool()
@friendly_errors
def find_citations(file_name: str) -> dict:
    """Citation discovery: the paper's reference list and any arXiv IDs it cites."""
    _, refs = body_and_references(split_sections(extract_text(_pdf(file_name))))
    references = split_references(refs)
    arxiv_ids = sorted(set(re.findall(r"(?:arXiv:?\s*|arxiv\.org/abs/)(\d{4}\.\d{4,5})", refs, re.I)))
    return {"total_references": len(references), "arxiv_ids": arxiv_ids, "references": references}


if __name__ == "__main__":
    mcp.run()   # stdio transport
