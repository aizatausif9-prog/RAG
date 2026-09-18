import os
import io
import hashlib

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
    .subtitle {font-size: 17px; opacity: 0.75; margin-bottom: 25px;}
    .answer-box {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🧠 DocuMind RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a PDF → build a FAISS knowledge base → ask questions grounded in your document.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Open-source embedding model; downloaded automatically on first run.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# PDF + RAG helper functions
# -----------------------------
def extract_pdf_text(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(pages):
    chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for i, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "page": page["page"],
                "chunk_id": i,
                "text": chunk,
            })

    return chunks


def build_faiss_index(chunks, model):
    texts = [item["text"] for item in chunks]

    # SentenceTransformer performs tokenization internally and converts
    # each chunk into a dense embedding vector.
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_faiss(question, index, chunks, model, top_k=5):
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = chunks[int(idx)].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def ask_groq(question, retrieved_chunks, model_name):
    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets "
            "or set it as an environment variable."
        )

    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']} | Relevance {item['score']:.3f}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are DocuMind, a careful document question-answering assistant.

Answer the user's question using ONLY the provided document context.
Do not invent facts or use outside knowledge.

Rules:
1. If the answer is clearly present in the context, answer it directly.
2. If the context does not contain enough information, say:
   "I couldn't find that information in the uploaded document."
3. When possible, mention the relevant page number(s).
4. Keep the answer clear and useful.
5. Do not pretend that retrieved text proves something it does not say.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer from the document context only.
"""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_tokens=900,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ RAG Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=5,
        help="Number of document chunks sent to the LLM as context.",
    )

    model_name = st.selectbox(
        "Groq chat model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 🔐 API key")
    if get_groq_client() is None:
        st.warning("GROQ_API_KEY is not configured.")
    else:
        st.success("Groq API key detected.")

    st.divider()
    st.caption(
        "Pipeline: PDF → text extraction → chunks → embeddings → FAISS → "
        "retrieval → Groq LLM"
    )


# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"],
    help="For this beginner-friendly version, upload a text-based PDF.",
)

if uploaded_file is None:
    st.info("Upload a PDF to create your private document knowledge base.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
current_hash = file_hash(pdf_bytes)

# Rebuild only when a different PDF is uploaded.
if st.session_state.get("pdf_hash") != current_hash:
    with st.spinner("🔎 Reading PDF and building your knowledge base..."):
        pages = extract_pdf_text(pdf_bytes)

        if not pages:
            st.error(
                "I couldn't extract readable text from this PDF. "
                "It may be scanned/image-only. Try a text-based PDF."
            )
            st.stop()

        chunks = build_chunks(pages)
        embedding_model = load_embedding_model()
        index = build_faiss_index(chunks, embedding_model)

        st.session_state.pdf_hash = current_hash
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.chat_history = []

    st.success(
        f"Knowledge base ready — {len(pages)} pages and {len(chunks)} chunks indexed."
    )

# -----------------------------
# Document stats
# -----------------------------
pages = st.session_state.pages
chunks = st.session_state.chunks
index = st.session_state.index
embedding_model = load_embedding_model()

col1, col2, col3 = st.columns(3)
col1.metric("📑 Pages", len(pages))
col2.metric("🧩 Chunks", len(chunks))
col3.metric("🔎 Vector dimension", index.d)

st.divider()

# -----------------------------
# Chat
# -----------------------------
st.subheader("💬 Ask your document")

for message in st.session_state.get("chat_history", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your PDF...")

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching the document..."):
            retrieved = search_faiss(
                question,
                index,
                chunks,
                embedding_model,
                top_k=top_k,
            )

        if not retrieved:
            answer = "I couldn't find relevant information in the uploaded document."
            st.markdown(answer)
        else:
            with st.spinner("🤖 Generating a grounded answer..."):
                try:
                    answer = ask_groq(question, retrieved, model_name)
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"Groq request failed: {exc}")
                    answer = "I couldn't generate an answer because the Groq request failed."

            with st.expander("🔎 View retrieved sources"):
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i} — Page {item['page']} — "
                        f"Similarity {item['score']:.3f}**"
                    )
                    st.write(item["text"])

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )
