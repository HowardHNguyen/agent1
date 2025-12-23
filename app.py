import os
import re
import uuid
import shutil
import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# Optional (only if you enable Elastic Cloud)
try:
    from elasticsearch import Elasticsearch
except Exception:
    Elasticsearch = None

# LLM (Groq via LangChain)
from langchain_groq import ChatGroq


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Advanced RAG Agent", layout="wide")

st.title("Advanced RAG Agent (Query Routing + Fusion Retrieval + Rerank + LLM)")


# -----------------------------
# Collapsible sections (Top)
# -----------------------------
with st.expander("How to Use (Quick Guide)", expanded=False):
    st.markdown(
        """
### Step-by-step

**1) Upload**
Upload one or more `.txt` documents (playbooks, SOPs, briefs, KPI definitions, FAQs, etc.).

**2) Index uploaded files (required)**
Click **Index uploaded files** to:
- chunk the text (word-safe),
- embed each chunk,
- store them in the vector index (ChromaDB).

> Uploading alone does not make documents searchable — indexing loads knowledge into the agent.

**3) Ask a question**
Ask something answerable from your docs, e.g.:
- “How should we structure a basic A/B test for a landing page?”
- “What is the difference between attribution and incrementality?”
- “How should we handle PII in marketing analytics and AI prompts?”

**4) Run Agent**
Click **Run Agent**. The app shows:
- the grounded answer,
- source evidence,
- optional technical details.

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
- **AEP / AJO / CJA:** use approved playbooks, KPI definitions, governance docs as the grounding layer for journeys/measurement
- **Salesforce:** index Knowledge/Case/FAQ content and enable consistent operational answers across teams
- **Databricks:** schedule ingestion + redaction + embedding jobs; run evaluations, guardrails, monitoring
- **Snowflake / CDPs:** govern source tables/views; attach metadata for access control and data boundaries

### Simple architecture diagram (for decks)
**User → Streamlit UI → Retriever (Vector / optional Lexical+Fusion) → Reranker → Context Builder → LLM (Groq) → Answer**
with **Source Evidence** and **Context Used** shown for trust/audit.
        """.strip()
    )


# -----------------------------
# Secrets / Keys
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    return st.secrets.get(name, os.environ.get(name, default))

GROQ_API_KEY = get_secret("GROQ_API_KEY")

# Optional Elastic Cloud secrets
ELASTIC_URL = get_secret("ELASTIC_URL")
ELASTIC_API_KEY = get_secret("ELASTIC_API_KEY")
ELASTIC_INDEX = get_secret("ELASTIC_INDEX", "documents")

if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY. Add it in Streamlit → Settings → Secrets, or as an environment variable.")


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
# Vector Store (Chroma)
# -----------------------------
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="rag_docs")
    return collection

collection = get_chroma_collection()


# -----------------------------
# Optional Elasticsearch
# -----------------------------
def get_es_client():
    if not Elasticsearch:
        return None
    if not ELASTIC_URL or not ELASTIC_API_KEY:
        return None
    try:
        return Elasticsearch(ELASTIC_URL, api_key=ELASTIC_API_KEY)
    except Exception:
        return None

es = get_es_client()


# -----------------------------
# Text utilities
# -----------------------------
def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 120):
    """
    Word-boundary safe chunker:
    - avoids starting or ending chunks in the middle of a word
    - preserves overlap for continuity
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    chunks = []
    n = len(text)
    start = 0

    while start < n:
        # Ensure chunk start is at a word boundary
        if start > 0 and text[start] != " " and text[start - 1] != " ":
            next_space = text.find(" ", start)
            if next_space == -1:
                break
            start = next_space + 1

        end = min(n, start + chunk_size)

        # Ensure chunk end is at a word boundary
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
        "movie": ["film", "cinema"],
        "heart attack": ["myocardial infarction", "MI"],
        "CVD": ["cardiovascular disease", "cardiac risk"]
    }
    extra_terms = []
    q_lower = q.lower()
    for k, syns in expansions.items():
        if k.lower() in q_lower:
            extra_terms.extend(syns)
    if extra_terms:
        q = q + " (" + " OR ".join(extra_terms) + ")"
    return q

def advanced_query_routing(query: str) -> str:
    q = query.lower()
    lexical_triggers = ["exact", "quote", "verbatim", "title", "named", "who is", "when is", "where is"]
    if any(t in q for t in lexical_triggers):
        return "lexical"
    return "vector"

def safe_preview(text: str, limit: int = 700) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit].rsplit(" ", 1)[0] + "..."


# -----------------------------
# Retrieval
# -----------------------------
def vector_retrieve(query: str, top_k: int):
    q_emb = embedder.encode([query])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = res.get("documents", [[]])[0]
    ids = res.get("ids", [[]])[0]
    return [{"id": ids[i], "text": docs[i], "source": "vector"} for i in range(len(docs))]

def lexical_retrieve(query: str, top_k: int):
    if not es:
        return []
    body = {"size": top_k, "query": {"match": {"content": query}}}
    try:
        r = es.search(index=ELASTIC_INDEX, body=body)
    except Exception as e:
        st.warning(f"Lexical retrieval unavailable (Elastic issue): {type(e).__name__}")
        return []

    hits = r.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        out.append({
            "id": h.get("_id", str(uuid.uuid4())),
            "text": h.get("_source", {}).get("content", ""),
            "source": "lexical"
        })
    return out

def fusion_retrieval(query: str, top_k: int):
    vec = vector_retrieve(query, top_k=top_k)
    lex = lexical_retrieve(query, top_k=top_k)
    combined = vec + lex
    seen = set()
    unique = []
    for item in combined:
        key = (item.get("text") or "")[:200]
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[: (top_k * 2)]

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

    if mode == "auto":
        route = advanced_query_routing(transformed)
    else:
        route = mode

    if route == "vector":
        retrieved = vector_retrieve(transformed, top_k=top_k)
    elif route == "lexical":
        retrieved = lexical_retrieve(transformed, top_k=top_k)
    else:
        retrieved = fusion_retrieval(transformed, top_k=top_k)

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
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 1500, 900, 50)
overlap = st.sidebar.slider("Overlap (chars)", 0, 400, 120, 20)

# ✅ Reset DB button (deletes chroma_db, clears caches, reruns)
if st.sidebar.button("Reset vector DB (delete index)"):
    shutil.rmtree("./chroma_db", ignore_errors=True)
    st.cache_resource.clear()
    st.sidebar.success("Vector DB reset. The app will reload — please re-index your files.")
    st.rerun()

if st.sidebar.button("Index uploaded files"):
    if not uploaded:
        st.sidebar.error("Please upload at least one .txt file.")
    else:
        added = 0
        for f in uploaded:
            text = f.read().decode("utf-8", errors="ignore")
            chunks = simple_chunk(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                continue

            ids = [str(uuid.uuid4()) for _ in chunks]
            embs = embedder.encode(chunks).tolist()
            collection.add(ids=ids, documents=chunks, embeddings=embs)
            added += len(chunks)

        st.sidebar.success(f"Indexed {added} chunks into Chroma.")


st.sidebar.header("2) Retrieval Settings")

# ✅ Turn off lexical/fusion in UI unless Elastic is truly available
retrieval_options = ["auto", "vector"]
if es:  # only show if configured & client created
    retrieval_options += ["lexical", "fusion"]

mode = st.sidebar.selectbox("Retrieval mode", retrieval_options)
top_k = st.sidebar.slider("Top-K", 2, 10, 5)

if not es:
    st.sidebar.caption("Lexical/Fusion are hidden (Elastic not configured).")

show_scores = st.sidebar.checkbox("Show technical relevance scores", value=False)


# -----------------------------
# Main: Executive-first Q&A UI
# -----------------------------
query = st.text_input("Ask a question")
run = st.button("RUN AGENT")

if run and query.strip():
    with st.spinner("Generating grounded answer..."):
        transformed, route, ranked, context, answer = advanced_rag_pipeline(query, mode=mode, top_k=top_k)

    # ✅ Answer FIRST and prominent
    st.subheader("Answer")
    st.caption("Grounded response generated from the indexed documents (with visible evidence below).")
    st.markdown(
        f"""
<div style="background:#f8fafc; padding:16px; border-radius:10px; border:1px solid #e5e7eb; font-size:1.05rem;">
{answer}
</div>
        """.strip(),
        unsafe_allow_html=True
    )

    # ✅ Evidence next (business-friendly)
    if ranked:
        with st.expander("Why this answer? (Source Evidence)", expanded=True):
            for i, d in enumerate(ranked, start=1):
                header = f"**Source {i}**"
                if show_scores:
                    header += f"  |  score={d['score']:.4f}  |  {d['source']}"
                st.markdown(header)
                st.markdown(safe_preview(d.get("text", ""), limit=700))
                st.markdown("---")

    # ✅ Technical details last (optional)
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
    unsafe_allow_html=True
)
