"""
The web app (MCP client side). Run:  streamlit run app.py

Upload a PDF -> Analyse -> ask questions -> compare -> citations.
Every button calls a tool on the MCP server through mcp_client.call_tool().
"""
import streamlit as st

import config
from mcp_client import call_tool

st.set_page_config(page_title="Research Paper Analyst", layout="wide")
st.title("Research Paper Analyst")
st.caption("MCP server + Gemma 3 (Ollama) + FAISS semantic search")

# ---------- Sidebar: upload + choose paper ----------
with st.sidebar:
    st.header("Papers")
    uploaded = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    if uploaded:
        (config.PAPERS_DIR / uploaded.name).write_bytes(uploaded.getvalue())
        st.success(f"Saved {uploaded.name}")

    try:
        papers = call_tool("list_papers")
    except Exception as exc:
        st.error(f"MCP server error: {exc}")
        st.stop()

    if not papers:
        st.info("Upload a PDF to begin.")
        st.stop()

    names = [p["file"] for p in papers]
    selected = st.selectbox("Paper", names)
    st.caption("Analysed" if next(p for p in papers if p["file"] == selected)["analysed"]
               else "Not analysed yet")

tab_analysis, tab_ask, tab_compare, tab_cite = st.tabs(
    ["Full analysis", "Ask a question", "Compare", "Citations"])

# ---------- Full analysis ----------
with tab_analysis:
    if st.button("Analyse paper", type="primary"):
        with st.spinner("Gemma 3 is reading the paper section by section (a few minutes the first time)..."):
            try:
                st.session_state[f"report:{selected}"] = call_tool("analyze_paper", {"file_name": selected})
            except Exception as exc:
                st.error(str(exc))
    report = st.session_state.get(f"report:{selected}")
    if report:
        st.markdown(report)
        st.download_button("Download report (.md)", report, file_name=f"{selected[:-4]}_analysis.md")

# ---------- Ask a question ----------
with tab_ask:
    question = st.text_input("Question about this paper",
                             placeholder="What dataset did they use and how big is it?")
    if st.button("Ask") and question:
        with st.spinner("Searching the paper and answering..."):
            try:
                result = call_tool("ask_paper", {"file_name": selected, "question": question})
                st.markdown(result["answer"])
                with st.expander("Passages used"):
                    for s in result["sources"]:
                        st.markdown(f"**[{s['n']}]** {s['section']} (similarity {s['score']}): {s['preview']}...")
            except Exception as exc:
                st.error(str(exc))

# ---------- Compare ----------
with tab_compare:
    others = [n for n in names if n != selected]
    if not others:
        st.info("Upload a second paper to compare.")
    else:
        other = st.selectbox("Compare with", others)
        if st.button("Compare papers"):
            with st.spinner("Comparing (analyses both papers first if needed)..."):
                try:
                    st.markdown(call_tool("compare_papers", {"file_a": selected, "file_b": other}))
                except Exception as exc:
                    st.error(str(exc))

# ---------- Citations ----------
with tab_cite:
    if st.button("Find citations"):
        try:
            cites = call_tool("find_citations", {"file_name": selected})
            st.metric("References found", cites["total_references"])
            if cites["arxiv_ids"]:
                st.write("arXiv papers cited:", ", ".join(cites["arxiv_ids"]))
            for n, ref in enumerate(cites["references"], 1):
                st.markdown(f"{n}. {ref}")
        except Exception as exc:
            st.error(str(exc))
