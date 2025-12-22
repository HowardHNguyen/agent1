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

with st.expander("How this works", expanded=False):
    st.write(
        "This agent transforms your query, routes retrieval (vector / lexical / fusion), "
        "reranks results for relevance, compresses the context, then generates an answer with an LLM."
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
    # Strong lightweight reranker
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
    # Persistent storage path (works on Streamlit Cloud; resets if app redeploys)
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
    # Keep your notebook’s idea, but make it practical:
    # - normalize whitespace
    # - expand a couple common synonyms (customize as you like)
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
    # Simple routing heuristic:
    # - lexical if user asks for exact quotes / exact titles / names
    # - vector otherwise
    q = query.lower()
    lexical_triggers = ["exact", "quote", "verbatim", "title", "named", "who is", "when is", "where is"]
    if any(t in q for t in lexical_triggers):
        return "lexical"
    return "vector"


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
        # ✅ Key fix: index missing or ES misconfigured → return empty instead of crashing
        st.warning(f"Lexical retrieval unavailable (Elastic issue): {type(e).__name__}")
        return []

    hits = r.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        out.append({
            "id": h.get("_id", str(uuid.uuid4())),
            "text": h["_source"].get("content", ""),
            "source": "lexical"
        })
    return out


def fusion_retrieval(query: str, top_k: int):
    # Combine and de-duplicate
    vec = vector_retrieve(query, top_k=top_k)
    lex = lexical_retrieve(query, top_k=top_k)
    combined = vec + lex
    seen = set()
    unique = []
    for item in combined:
        key = item["text"][:200]
        if key not in seen:
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
    # Simple “compression”: take top docs until we hit the budget.
    context_parts = []
    total = 0
    for d in ranked_docs:
        t = d["text"].strip()
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

# Show lexical/fusion ONLY if Elastic is configured
retrieval_options = ["auto", "vector"]
if es:  # Elastic client exists only when ELASTIC_URL + ELASTIC_API_KEY are present
    retrieval_options += ["lexical", "fusion"]

mode = st.sidebar.selectbox("Retrieval mode", retrieval_options)
top_k = st.sidebar.slider("Top-K", 2, 10, 5)

# Optional: if Elastic is missing, keep a helpful hint (but user won't see lexical/fusion anyway)
if (not es):
    st.sidebar.caption("Tip: Elastic (lexical/fusion) is hidden because ELASTIC_URL / ELASTIC_API_KEY are not set.")



# -----------------------------
# Main Chat UI
# -----------------------------
query = st.text_input("Ask a question")

colA, colB = st.columns([1, 1])

if st.button("Run Agent") and query.strip():
    with st.spinner("Retrieving + reranking + generating answer..."):
        transformed, route, ranked, context, answer = advanced_rag_pipeline(query, mode=mode, top_k=top_k)

    st.subheader("Answer")
    st.write(answer)

    with colA:
        st.subheader("Pipeline Details")
        st.write(f"**Transformed query:** {transformed}")
        st.write(f"**Route:** {route}")

    with colB:
        st.subheader("Top Reranked Chunks")
        for i, d in enumerate(ranked, start=1):
            st.write(f"**#{i} | score={d['score']:.4f} | source={d['source']}**")
            st.write(d["text"][:700] + ("..." if len(d["text"]) > 700 else ""))

    with st.expander("Context used for generation"):
        st.text(context)
