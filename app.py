import os
import re
import uuid
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

# --- Collapsible sections ---
with st.expander("How to Use (Quick Guide)", expanded=False):
    st.markdown(
        """
### Step-by-step

**1) Upload**
Upload one or more `.txt` documents (marketing playbooks, SOPs, briefs, KPI definitions, FAQs, etc.).

**2) Index uploaded files (required)**
Click **Index uploaded files** to:
- chunk the text,
- embed each chunk,
- store them in the vector index (ChromaDB).

> Uploading alone does not make documents searchable — indexing is what “loads” knowledge into the agent.

**3) Ask a question**
Ask something that can be answered from your documents, for example:
- “How should we structure a basic A/B test for a landing page?”
- “What is the difference between attribution and incrementality?”
- “How should we handle PII in marketing analytics and AI prompts?”

**4) Run Agent**
Click **Run Agent**. The app will display:
- the final grounded answer,
- source evidence (retrieved chunks),
- optional technical details.

**Tip**
If the agent says **No documents were retrieved**, it usually means indexing hasn’t happened yet
(or the index is empty).
        """.strip()
    )

with st.expander("About This Agent", expanded=False):
    st.markdown(
        """
### Models & Technologies Used

This AI agent is built using a **modern, modular AI stack** designed to mirror how
production-grade AI systems are implemented in enterprise environments.

#### Language Model (LLM)
- **Groq-hosted LLM (LLaMA family)**  
  Used exclusively for **final answer synthesis**, not retrieval.
- The LLM is constrained to generate responses **only from retrieved context** to reduce hallucinations.

#### Embedding Model
- **Sentence-Transformers (Transformer embeddings)**  
  Converts document chunks and user queries into dense vectors for semantic search.

#### Vector Database
- **ChromaDB** (local-first vector store)  
  Easily swappable with enterprise vector DBs (Pinecone, Weaviate, OpenSearch, etc.).

#### Reranking Model
- **Cross-Encoder reranker**  
  Improves precision by jointly scoring (query, chunk) pairs.

#### Query Routing & Retrieval Strategy
- **Auto-routing** chooses a retrieval strategy (vector by default).
- Architecture supports lexical/hybrid retrieval when enabled (e.g., Elastic).

#### Context Management
- **Context compression** selects only the most relevant passages to fit model limits and reduce noise.

### Engineering Principles Demonstrated
- **Separation of concerns** (retrieval, ranking, generation are modular)
- **Explainability by design** (evidence is visible)
- **Safety-first AI usage** (avoid putting raw PII into prompts)
- **Enterprise extensibility** (components can be upgraded independently)
        """.strip()
    )


# -----------------------------
# Secrets / Keys
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    # Streamlit secrets first, then env var
    return st.secrets.get(name, os.environ.get(name, default))

GROQ_API_KEY = get_secret("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY. Add it in Streamlit → Settings → Secrets, or as an environment variable.")

# Optional Elastic Cloud secrets
ELASTIC_URL = get_secret("ELASTIC_URL")
ELASTIC_API_KEY = get_secret("ELASTIC_API_KEY")  # preferred over basic auth
ELASTIC_INDEX = get_secret("ELASTIC_INDEX", "documents")


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
# Text utilities
# -----------------------------
def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 120):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
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
    """Word-safe preview: avoids chopping the first character, preserves whole words."""
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    # keep whole words
    return t[:limit].rsplit(" ", 1)[0] + "..."


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
# Sidebar: Ingest Docs
# -----------------------------
st.sidebar.header("1) Add / Index Documents")

uploaded = st.sidebar.file_uploader(
    "Upload .txt files (simple first version).",
    type=["txt"],
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 1500, 900, 50)
overlap = st.sidebar.slider("Overlap (chars)", 0, 400, 120, 20)

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

retrieval_options = ["auto", "vector"]
if es:
    retrieval_options += ["lexical", "fusion"]

mode = st.sidebar.selectbox("Retrieval mode", retrieval_options)
top_k = st.sidebar.slider("Top-K", 2, 10, 5)

if not es:
    st.sidebar.caption("Tip: Elastic (lexical/fusion) is hidden because ELASTIC_URL / ELASTIC_API_KEY are not set.")

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
            st.text(context)


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
