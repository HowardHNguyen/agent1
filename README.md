# Advanced RAG Agent (Streamlit)

This app implements an advanced Retrieval-Augmented Generation (RAG) pipeline:
- Query transformation
- Query routing (auto / vector / lexical / fusion)
- Vector retrieval (ChromaDB)
- Optional lexical retrieval (Elastic Cloud)
- Reranking (CrossEncoder)
- Context compression
- Answer generation (Groq LLM via LangChain)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
