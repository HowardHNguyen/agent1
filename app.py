import os
import re
import uuid
import tempfile
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings as ChromaSettings

# LLM (Groq via LangChain)
from langchain_groq import ChatGroq


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Advanced RAG Agent", layout="wide")
st.title("Advanced RAG Agent (Query Routing + Fusion Retrieval + Rerank + LLM)")


# -----------------------------
# Secrets / Keys
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    return st.secrets.get(name, os.environ.get(name, default))


GROQ_API_KEY = get_secret("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY. Add it in Streamlit → Settings → Secrets, or as an environment variable.")


# -----------------------------
# Collapsible sections (Top)
# -----------------------------
with st.expander("How to Use (Quick Guide)", expanded=False):
    st.markdown(
        """
### Step-by-step

**This demo shows how an AI Agent ingests knowledge, retrieves relevant information, and generates grounded answers.**

**Step 1 - Upload documents**

Upload one or more `.txt` files (e.g., marketing playbooks, experiment frameworks, KPI definitions, governance docs).

**Step 2 - Index uploaded files (required)**

Click **Index uploaded files** to:
- split documents into semantic chunks,
- generate embeddings,
- store them in the vector database (ChromaDB).

> Uploading alone does not make documents searchable — indexing loads knowledge into the agent. This step loads knowledge into the agent so it can answer questions.

✅ You only need to index again when documents changes.

**Step 3 - (Optional) Reset the knowledge base**

Click **Reset vector DB (delete index)** when you want to:
- completely clear the indexed knowledge,
- start fresh with a new set of documents,
- avoid mixing old and new content.

⚠️ This permanently deletes the current index and requires re-indexing.

**Step 4 - Ask a question**

Enter a natural-language question based on the uploaded content, such as:
- “How should we structure a basic A/B test for a landing page?”
- “What is the difference between attribution and incrementality?”
- “How should we handle PII in marketing analytics and AI prompts?”

**Step 5 - Run the Agent**

Click **RUN AGENT** to:
- retrieve the most relevant document sections,
- rerank them for precision,
- generate a grounded response using an LLM.

The app displays:
- the **final answer** (for business review),
- **source evidence** (for trust and transparency),
- optional **technical details** (for engineering review).

**Tip**
If you see **No documents were retrieved**, indexing hasn’t happened yet (or the index is empty).
        """.strip()
    )

with st.expander("About This Agent", expanded=False):
    st.markdown(
        """
### What this demo shows
A production-style **Retrieval-Augmented Generation (RAG) AI Agent**: retrieve relevant content from your indexed
documents, rerank for precision, then generate a grounded response with an LLM.

### Models & Technologies
- **LLM:** Groq-hosted LLM (LLaMA family) for final answer synthesis
- **Embeddings:** Sentence-Transformers for semantic search
- **Vector DB:** ChromaDB (local-first; swappable with enterprise vector DBs)
- **Reranker:** Cross-Encoder reranker (query–chunk pair scoring)

### Why this scales to enterprise
This architecture scales because it separates **ingestion**, **retrieval**, **ranking**, and **generation** into modular,
governable components. It starts as a fast prototype (Streamlit + local vector DB) and evolves into production by swapping
in enterprise equivalents (managed vector search, centralized logging/monitoring, RBAC/ABAC, scheduled indexing pipelines).
It supports compliance workflows by keeping the LLM grounded in retrieved context and enabling auditability via visible
source evidence and generation context.

### How this integrates with MarTech / Data stacks
- **AEP / AJO / CJA:** index playbooks, KPI definitions, governance docs; ground measurement and journey decisions
- **Salesforce:** index Knowledge/Case/FAQ content and enable consistent operational answers across teams
- **Databricks:** schedule ingestion + redaction + embedding jobs; run evaluations, guardrails, monitoring
- **Snowflake / CDPs:** govern source tables/views; attach metadata for access control and boundaries

### Simple architecture diagram (for decks)
**User → Streamlit UI → Retriever (Vector) → Reranker → Context Builder → LLM (Groq) → Answer**
with **Source Evidence** and **Context Used** shown for trust/audit.
        """.strip()
    )


# -----------------------------
# Models (cached)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource
def load_llm():
    if not GROQ_API_KEY:
        return None
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)


embedder = load_embedder()
reranker = load_reranker()
llm = load_llm()


# -----------------------------
# Chroma (Streamlit Cloud: MUST write to /tmp)
# -----------------------------
CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")
CHROMA_COLLECTION = "rag_docs"


def get_chroma_client():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
    return chromadb.PersistentClient(path=CHROMA_PATH, settings=settings)


def get_chroma_collection():
    client = get_chroma_client()
    col = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return client, col


# -----------------------------
# Text utilities
# -----------------------------
def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 120):
    """
    Word-boundary safe chunker
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    chunks = []
    n = len(text)
    start = 0

    while start < n:
        if start > 0 and text[start] != " " and text[start - 1] != " ":
            next_space = text.find(" ", start)
            if next_space == -1:
                break
            start = next_space + 1

        end = min(n, start + chunk_size)

        if end < n and text[end] != " ":
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def advanced_query_transformation(query: str) -> str:
    q = re.sub(r"\s+", " ", query).strip()
    expansions = {
        "heart attack": ["myocardial infarction", "MI"],
        "CVD": ["cardiovascular disease", "cardiac risk"],
    }
    extra_terms = []
    q_lower = q.lower()
    for k, syns in expansions.items():
        if k.lower() in q_lower:
            extra_terms.extend(syns)
    if extra_terms:
        q = q + " (" + " OR ".join(extra_terms) + ")"
    return q


def safe_preview(text: str, limit: int = 700) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rsplit(" ", 1)[0] + "..."


# -----------------------------
# Chroma: batched upsert helpers
# -----------------------------
def batched(lst, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def chroma_upsert_batched(collection, docs: list[str], batch_size_docs: int = 64) -> int:
    if not docs:
        return 0
    total = 0
    for doc_batch in batched(docs, batch_size_docs):
        ids = [str(uuid.uuid4()) for _ in doc_batch]
        embeddings = embedder.encode(doc_batch).tolist()
        collection.upsert(ids=ids, documents=doc_batch, embeddings=embeddings)
        total += len(doc_batch)
    return total


# -----------------------------
# Retrieval (Vector only)
# -----------------------------
def vector_retrieve(query: str, top_k: int):
    _, collection = get_chroma_collection()
    q_emb = embedder.encode([query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    ids = res.get("ids", [[]])[0]
    return [{"id": ids[i], "text": docs[i], "source": "vector"} for i in range(len(docs))]


def rerank_documents(query: str, docs: list, top_k: int):
    if not docs:
        return []
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [{"score": float(s), **d} for d, s in ranked[:top_k]]


def select_and_compress_context(ranked_docs: list, max_chars: int = 3000):
    context_parts = []
    total = 0
    for d in ranked_docs:
        t = (d.get("text") or "").strip()
        if not t:
            continue
        if total + len(t) > max_chars:
            t = t[: max(0, max_chars - total)]
        context_parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, context: str):
    if not llm:
        return "LLM is not configured. Please add GROQ_API_KEY in Streamlit Secrets."

    prompt = (
        "You are a helpful assistant. Answer the user using ONLY the provided context.\n"
        "If the context is insufficient, say what is missing and ask a clarifying question.\n\n"
        f"USER QUERY:\n{query}\n\n"
        f"CONTEXT:\n{context}\n"
    )
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Groq call failed: {type(e).__name__}: {str(e)}"


def advanced_rag_pipeline(query: str, mode: str, top_k: int):
    transformed = advanced_query_transformation(query)

    # Even if mode is "auto", we keep it simple for exec demo: Vector retrieval only
    route = "vector"

    retrieved = vector_retrieve(transformed, top_k=top_k)

    if not retrieved:
        return transformed, route, [], "", (
            "No documents were retrieved. "
            "Please click **Index uploaded files** first, then ask your question again."
        )

    ranked = rerank_documents(query, retrieved, top_k=top_k)
    context = select_and_compress_context(ranked)
    answer = generate_answer(query, context)
    return transformed, route, ranked, context, answer


# -----------------------------
# Sidebar: Ingest Docs + Reset
# -----------------------------
st.sidebar.header("1) Add / Index Documents")

uploaded = st.sidebar.file_uploader(
    "Upload .txt files (simple first version).",
    type=["txt"],
    accept_multiple_files=True,
)

chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 1500, 900, 50)
overlap = st.sidebar.slider("Overlap (chars)", 0, 400, 120, 20)

if st.sidebar.button("Reset vector DB (delete index)"):
    try:
        client = get_chroma_client()
        client.reset()
        st.sidebar.success("Vector DB reset. Please re-index your files.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Reset failed: {type(e).__name__}")
        st.sidebar.caption(str(e))

if st.sidebar.button("Index uploaded files"):
    if not uploaded:
        st.sidebar.error("Please upload at least one .txt file.")
    else:
        try:
            all_chunks = []
            for f in uploaded:
                text = f.read().decode("utf-8", errors="ignore")
                chunks = simple_chunk(text, chunk_size=chunk_size, overlap=overlap)
                all_chunks.extend([c for c in chunks if c and len(c.strip()) > 0])

            if not all_chunks:
                st.sidebar.error("No valid text chunks found. Please upload a non-empty .txt.")
            else:
                _, collection = get_chroma_collection()
                added = chroma_upsert_batched(collection, all_chunks, batch_size_docs=64)
                st.sidebar.success(f"Indexed {added} chunks into Chroma (stored in /tmp).")
        except Exception as e:
            st.sidebar.error(f"Indexing failed: {type(e).__name__}")
            st.sidebar.caption(str(e))


# -----------------------------
# Retrieval Settings (ONLY auto + vector)
# -----------------------------
st.sidebar.header("2) Retrieval Settings")
mode = st.sidebar.selectbox("Retrieval mode", ["auto", "vector"])
top_k = st.sidebar.slider("Top-K", 2, 10, 5)
show_scores = st.sidebar.checkbox("Show technical relevance scores", value=False)


# -----------------------------
# Main: Executive-first Q&A UI
# -----------------------------
query = st.text_input("Ask a question")
run = st.button("RUN AGENT")

if run and query.strip():
    with st.spinner("Generating grounded answer..."):
        transformed, route, ranked, context, answer = advanced_rag_pipeline(query, mode=mode, top_k=top_k)

    st.subheader("Answer")
    st.caption("Grounded response generated from the indexed documents (with visible evidence below).")

    st.markdown(
        f"""
<div style="background:#f8fafc; padding:16px; border-radius:10px; border:1px solid #e5e7eb; font-size:1.05rem; margin-bottom:14px;">
{answer}
</div>
        """.strip(),
        unsafe_allow_html=True,
    )

    # ✅ extra spacing so Answer doesn't look glued to the expander
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    if ranked:
        with st.expander("Why this answer? (Source Evidence)", expanded=True):
            for i, d in enumerate(ranked, start=1):
                header = f"**Source {i}**"
                if show_scores:
                    header += f"  |  score={d['score']:.4f}  |  {d['source']}"
                st.markdown(header)
                st.markdown(safe_preview(d.get("text", ""), limit=700))
                st.markdown("---")

    with st.expander("Technical Details (for engineering review)", expanded=False):
        st.write(f"**Transformed query:** {transformed}")
        st.write(f"**Retrieval route:** {route}")
        st.write(f"**Top-K:** {top_k}")
        with st.expander("Context used for generation", expanded=False):
            st.text_area("Context", value=context, height=260)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
<div style="text-align:center; font-size: 0.9rem; line-height: 1.6; color: #6b7280;">
  <div><b>Advanced RAG Agent</b> — Executive Demo for AI/ML + MarTech Innovation</div>
  <div>Built by Howard Nguyen • Streamlit • ChromaDB • SentenceTransformers • Cross-Encoder Rerank • Groq LLM</div>
  <div style="margin-top:6px;">
    <span>⚠️ Demo app: answers are generated from uploaded documents and may require human review.</span>
  </div>
  <div style="margin-top:6px;">
    <span>Privacy note: avoid uploading sensitive PII unless you have approval and proper controls.</span>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)
