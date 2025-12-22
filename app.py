# app.py - Advanced RAG AI Agent for Movie Suggestions

import streamlit as st
import os
import chromadb
from chromadb import Client
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from langchain_groq import ChatGroq
from elasticsearch import Elasticsearch
from huggingface_hub import login
import torch.nn.functional as F

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="Advanced RAG Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Advanced RAG Movie Recommender")
st.markdown("Ask for movie suggestions — powered by fusion retrieval, reranking, and Groq's fast LLMs!")

# ============================
# Secure API Key Loading
# ============================
def load_api_keys():
    """Load Hugging Face and Groq API keys from multiple possible sources."""
    
    hf_token = None
    groq_key = None

    # 1. Try Streamlit secrets (recommended for streamlit.app)
    try:
        hf_token = st.secrets["HUGGING_FACE_API_KEY"]
        groq_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API keys loaded from Streamlit secrets.")
        return hf_token, groq_key
    except:
        pass

    # 2. Try Google Colab secrets
    try:
        from google.colab import userdata
        hf_token = userdata.get('HUGGING_FACE_API_KEY')
        groq_key = userdata.get('GROQ_API_KEY')
        st.success("✅ API keys loaded from Colab secrets.")
        return hf_token, groq_key
    except:
        pass

    # 3. Try local keys.txt file (for local testing)
    if os.path.exists("keys.txt"):
        with open("keys.txt", "r") as f:
            lines = f.read().strip().splitlines()
            for line in lines:
                if line.startswith("HUGGING_FACE_API_KEY"):
                    hf_token = line.split("=")[1].strip().strip('"').strip("'")
                elif line.startswith("GROQ_API_KEY"):
                    groq_key = line.split("=")[1].strip().strip('"').strip("'")
        if hf_token and groq_key:
            st.success("✅ API keys loaded from keys.txt file.")
            return hf_token, groq_key

    return None, None

hf_api_token, groq_api_key = load_api_keys()

if not hf_api_token or not groq_api_key:
    st.error("❌ Missing API keys!")
    st.info(
        "Please provide your keys using one of these methods:\n"
        "- Add them in **Streamlit Secrets** (recommended for deployment)\n"
        "- Add them in **Colab Secrets** (if running in Colab)\n"
        "- Create a `keys.txt` file in the same folder with:\n"
        "```\n"
        "HUGGING_FACE_API_KEY=hf_...\n"
        "GROQ_API_KEY=gsk_...\n"
        "```"
    )
    st.stop()

# Login to Hugging Face
login(token=hf_api_token)
os.environ["GROQ_API_KEY"] = groq_api_key

# ============================
# Load Models (Cached)
# ============================
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... This may take 1-2 minutes on first run."):
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        rerank_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        rerank_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return sentence_model, rerank_tokenizer, rerank_model, summarizer

sentence_model, rerank_tokenizer, rerank_model, summarizer = load_models()

# LLM Setup
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.7,
        max_tokens=1024,
    )

llm = get_llm()

# ============================
# Setup Vector DB (Chroma) & Elasticsearch
# ============================
@st.cache_resource
def setup_databases():
    # ChromaDB
    client = chromadb.Client()
    collection_name = "movies"
    try:
        collection = client.create_collection(name=collection_name)
    except:
        collection = client.get_collection(name=collection_name)

    # Sample movie documents
    sample_docs = [
        "The Shawshank Redemption is a deeply moving prison drama about hope and friendship. Perfect for a thoughtful rainy day.",
        "Forrest Gump takes you on an emotional journey through decades of American history with incredible heart.",
        "Inception is a mind-bending sci-fi thriller by Christopher Nolan — great if you love complex plots.",
        "The Godfather is a masterpiece of crime drama, family, and power. A must-watch classic.",
        "Spirited Away is a beautiful animated fantasy adventure from Studio Ghibli — magical and uplifting."
    ]

    if collection.count() == 0:
        collection.add(
            documents=sample_docs,
            ids=[f"doc_{i}" for i in range(len(sample_docs))]
        )

    # Elasticsearch (using public demo or skip if not available)
    # Note: Your original ES instance may not be accessible publicly → we skip critical dependency
    es = None
    try:
        es = Elasticsearch(
            hosts=['https://2e8be0fdffac440795fcfbecf86079b4.us-central1.gcp.cloud.es.io'],
            http_auth=('elastic', 'aPRfFBGj1FkOMgvm7XqkgAFJ')
        )
        if es.ping():
            st.success("Connected to Elasticsearch")
        else:
            es = None
    except:
        es = None

    return collection, es

collection, es_client = setup_databases()

# ============================
# RAG Pipeline Functions
# ============================
def fusion_retrieval(query, top_k=5):
    # Vector search (Chroma)
    query_emb = sentence_model.encode([query])
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=top_k
    )
    vector_docs = results['documents'][0]

    # Keyword search (Elasticsearch fallback)
    es_docs = []
    if es_client:
        try:
            res = es_client.search(
                index="movies",
                body={"query": {"match": {"content": query}}, "size": top_k}
            )
            es_docs = [hit["_source"]["content"] for hit in res['hits']['hits']]
        except:
            pass

    # Combine and deduplicate
    all_docs = list(dict.fromkeys(vector_docs + es_docs))
    return all_docs[:top_k]

def rerank_documents(query, documents):
    if not documents:
        return []
    inputs = [rerank_tokenizer(query, doc, truncation=True, padding=True, max_length=512, return_tensors="pt") for doc in documents]
    scores = []
    for inp in inputs:
        with torch.no_grad():
            outputs = rerank_model(**inp)
            probs = F.softmax(outputs.logits, dim=1)
            scores.append(probs[0][1].item())  # Relevance score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

def compress_context(documents):
    summaries = []
    for doc in documents:
        if len(doc.split()) > 30:
            try:
                summary = summarizer(doc, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except:
                summaries.append(doc[:200] + "...")
        else:
            summaries.append(doc)
    return summaries

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""[INST]
You are a friendly and expert movie recommender. Use the provided context to give helpful, engaging, and accurate movie suggestions.

User Question: {query}

Relevant Movie Info:
{context}

Provide a natural, detailed response with movie recommendations and brief reasons why they fit.
[/INST]"""

    response = llm.invoke(prompt)
    return response.content

def advanced_rag_pipeline(query):
    docs = fusion_retrieval(query)
    ranked = rerank_documents(query, docs)
    compressed = compress_context(ranked)
    answer = generate_answer(query, compressed)
    return answer

# ============================
# Main UI
# ============================
query = st.text_input(
    "🎥 What kind of movie are you in the mood for?",
    placeholder="e.g., good movies for a rainy day, uplifting feel-good films, mind-bending sci-fi..."
)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Searching movies and generating recommendations..."):
        try:
            answer = advanced_rag_pipeline(query.strip())
            st.markdown("### 🍿 Your Movie Recommendations")
            st.write(answer)
        except Exception as e:
            st.error(f"Oops! Something went wrong: {str(e)}")
elif st.button("Get Recommendations") and not query.strip():
    st.warning("Please enter a movie mood or question!")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Powered by Groq, Hugging Face, ChromaDB • Sample demo with small movie knowledge base")