"""
RAG Chatbot with LlamaParse + Streamlit
----------------------------------------
Features:
 - User login (simple username/password, stored in session)
 - Dark / Light theme toggle
 - Multiple PDF upload + parsing via LlamaParse
 - View parsed markdown/text per PDF
 - Chunking (RecursiveCharacterTextSplitter) + chunk viewer
 - Embeddings + Chroma vector store (per session, in-memory/persisted to disk)
 - Retrieval viewer (see which chunks were retrieved for a question)
 - Chat with RAG pipeline, answers grounded only in uploaded PDFs

Run:
    pip install -r requirements.txt
    streamlit run app.py

Required environment variables (set in a local .env file, NOT committed):
    LLAMA_PARSE_API_KEY   -> from https://cloud.llamaindex.ai
    GROQ_API_KEY          -> from https://console.groq.com
    (Embeddings run locally via sentence-transformers — no key needed.)
"""

import os
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads LLAMA_PARSE_API_KEY and GROQ_API_KEY from a local .env file

# ---------------------------------------------------------------------------
# Page config (must be first st call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports of heavy libs (so the app still boots even if a package
# is momentarily missing, giving a friendly error instead of a crash)
# ---------------------------------------------------------------------------
try:
    from llama_parse import LlamaParse
except ImportError:
    LlamaParse = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

try:
    from groq import Groq
except ImportError:
    Groq = None


# ---------------------------------------------------------------------------
# ---------------------------  AUTH (simple)  --------------------------------
# ---------------------------------------------------------------------------
# Demo users. Replace with a real DB / OAuth in production.
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "demo": hashlib.sha256("demo123".encode()).hexdigest(),
}


def check_login(username: str, password: str) -> bool:
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed


def login_screen():
    st.markdown(
        "<h1 style='text-align:center;'>🔐 PDF RAG Chatbot</h1>"
        "<p style='text-align:center;color:gray;'>Sign in to continue</p>",
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption("Demo credentials → admin / admin123  or  demo / demo123")


# ---------------------------------------------------------------------------
# --------------------------  THEME (dark/light)  ----------------------------
# ---------------------------------------------------------------------------
def inject_theme(theme: str):
    if theme == "Dark":
        bg, fg, card, accent, border = "#0e1117", "#fafafa", "#161b22", "#6366f1", "#30363d"
    else:
        bg, fg, card, accent, border = "#ffffff", "#1a1a1a", "#f5f5f7", "#6366f1", "#e2e2e2"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {fg};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {card};
            border-right: 1px solid {border};
        }}
        .chunk-card {{
            background-color: {card};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
            font-size: 0.85rem;
        }}
        .chunk-meta {{
            color: {accent};
            font-weight: 600;
            font-size: 0.75rem;
            margin-bottom: 4px;
        }}
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid {accent};
        }}
        .stButton>button:hover {{
            background-color: {accent};
            color: white;
        }}
        div[data-testid="stChatMessage"] {{
            background-color: {card};
            border-radius: 10px;
            border: 1px solid {border};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# -----------------------------  PARSING  ------------------------------------
# ---------------------------------------------------------------------------
def parse_pdf_with_llamaparse(file_bytes: bytes, filename: str, api_key: str) -> str:
    """Parse a single PDF using LlamaParse and return markdown text."""
    if LlamaParse is None:
        raise RuntimeError("llama-parse is not installed. Run: pip install llama-parse")

    os.makedirs("tmp_uploads", exist_ok=True)
    tmp_path = os.path.join("tmp_uploads", filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    parser = LlamaParse(api_key=api_key, result_type="markdown", verbose=False)
    documents = parser.load_data(tmp_path)
    text = "\n\n".join(d.text for d in documents)
    return text


# ---------------------------------------------------------------------------
# -----------------------------  CHUNKING  ------------------------------------
# ---------------------------------------------------------------------------
def chunk_text(text: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError("langchain is not installed. Run: pip install langchain")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [
        {"id": f"{source}::chunk-{i}", "source": source, "text": p, "index": i}
        for i, p in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# -------------------------  VECTOR STORE (Chroma)  ---------------------------
# ---------------------------------------------------------------------------
def get_chroma_collection():
    """Vector store using a free local embedding model (no API key needed)."""
    if chromadb is None:
        raise RuntimeError("chromadb is not installed. Run: pip install chromadb")

    client = chromadb.Client()  # in-memory, resets each session
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        client.delete_collection("pdf_rag")
    except Exception:
        pass
    collection = client.create_collection(name="pdf_rag", embedding_function=embed_fn)
    return collection


def add_chunks_to_collection(collection, chunks):
    if not chunks:
        return
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "index": c["index"]} for c in chunks],
    )


def retrieve(collection, query: str, k: int = 4):
    res = collection.query(query_texts=[query], n_results=k)
    retrieved = []
    if res["documents"]:
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            retrieved.append({"text": doc, "source": meta["source"], "index": meta["index"], "score": 1 - dist})
    return retrieved


# ---------------------------------------------------------------------------
# -------------------------------  LLM CALL  ----------------------------------
# ---------------------------------------------------------------------------
def call_llm(question: str, context_chunks, groq_api_key: str) -> str:
    if Groq is None:
        raise RuntimeError("groq is not installed. Run: pip install groq")

    client = Groq(api_key=groq_api_key)
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | chunk {c['index']}]\n{c['text']}" for c in context_chunks
    )
    system_prompt = (
        "You are a helpful assistant that answers questions strictly using the "
        "provided PDF context. If the answer is not in the context, say you "
        "could not find it in the uploaded documents. Always mention which "
        "source file(s) you used."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# --------------------------------  STATE  ------------------------------------
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "logged_in": False,
        "username": None,
        "theme": "Dark",
        "parsed_docs": {},      # filename -> parsed text
        "chunks": [],           # list of chunk dicts
        "collection": None,
        "messages": [],         # chat history
        "last_retrieved": [],   # last retrieval results
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# -----------------------------  MAIN APP  ------------------------------------
# ---------------------------------------------------------------------------
def main_app():
    inject_theme(st.session_state.theme)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, **{st.session_state.username}**")
        st.session_state.theme = st.radio("Theme", ["Dark", "Light"], horizontal=True,
                                           index=0 if st.session_state.theme == "Dark" else 1)
        st.divider()

        st.markdown("#### 🔑 API Keys")
        st.caption("Loaded from .env — override below if needed.")
        llama_key = st.text_input("LlamaParse API Key", type="password",
                                   value=os.environ.get("LLAMA_PARSE_API_KEY", ""))
        groq_key = st.text_input("Groq API Key", type="password",
                                  value=os.environ.get("GROQ_API_KEY", ""))
        st.caption("Embeddings run locally (sentence-transformers) — no key needed.")
        st.divider()

        st.markdown("#### 📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
        )

        col_a, col_b = st.columns(2)
        chunk_size = col_a.number_input("Chunk size", 200, 4000, 1000, step=100)
        chunk_overlap = col_b.number_input("Overlap", 0, 1000, 150, step=50)

        process_btn = st.button("🚀 Parse + Index PDFs", use_container_width=True)
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.title("📄 PDF RAG Chatbot")
    st.caption("Upload PDFs → parse with LlamaParse → chunk → retrieve → chat, all grounded in your documents.")

    # ---------- Processing pipeline ----------
    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        elif not llama_key:
            st.warning("Please provide a LlamaParse API key.")
        elif not groq_key:
            st.warning("Please provide a Groq API key (used for chat answers).")
        else:
            all_chunks = []
            progress = st.progress(0, text="Starting...")
            for i, file in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Parsing {file.name} ...")
                try:
                    text = parse_pdf_with_llamaparse(file.getvalue(), file.name, llama_key)
                    st.session_state.parsed_docs[file.name] = text
                    chunks = chunk_text(text, file.name, chunk_size, chunk_overlap)
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"Failed to parse {file.name}: {e}")
            progress.progress(1.0, text="Building vector index...")

            try:
                collection = get_chroma_collection()
                add_chunks_to_collection(collection, all_chunks)
                st.session_state.collection = collection
                st.session_state.chunks = all_chunks
                st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} PDF(s).")
            except Exception as e:
                st.error(f"Failed to build vector index: {e}")
            progress.empty()

    # ---------- Tabs ----------
    tab_chat, tab_parsed, tab_chunks, tab_retrieval = st.tabs(
        ["💬 Chat", "📜 Parsed PDFs", "🧩 Chunks", "🔍 Retrieved Context"]
    )

    # --- Chat tab ---
    with tab_chat:
        if not st.session_state.collection:
            st.info("Upload and process PDFs from the sidebar to start chatting.")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            question = st.chat_input("Ask something about your uploaded PDFs...")
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving relevant chunks..."):
                        retrieved = retrieve(st.session_state.collection, question, k=4)
                        st.session_state.last_retrieved = retrieved
                    with st.spinner("Generating answer..."):
                        try:
                            answer = call_llm(question, retrieved, groq_key)
                        except Exception as e:
                            answer = f"Error generating answer: {e}"
                    st.markdown(answer)
                    with st.expander("📎 Sources used"):
                        for r in retrieved:
                            st.caption(f"{r['source']} · chunk {r['index']} · score {r['score']:.2f}")

                st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Parsed PDFs tab ---
    with tab_parsed:
        if not st.session_state.parsed_docs:
            st.info("No PDFs parsed yet.")
        else:
            doc_name = st.selectbox("Select a document", list(st.session_state.parsed_docs.keys()))
            st.text_area("Parsed content", st.session_state.parsed_docs[doc_name], height=500)

    # --- Chunks tab ---
    with tab_chunks:
        if not st.session_state.chunks:
            st.info("No chunks yet. Process PDFs first.")
        else:
            sources = sorted(set(c["source"] for c in st.session_state.chunks))
            filter_src = st.selectbox("Filter by source", ["All"] + sources)
            shown = [c for c in st.session_state.chunks if filter_src == "All" or c["source"] == filter_src]
            st.caption(f"Showing {len(shown)} chunks")
            for c in shown:
                st.markdown(
                    f"<div class='chunk-card'><div class='chunk-meta'>{c['source']} · chunk {c['index']}</div>{c['text']}</div>",
                    unsafe_allow_html=True,
                )

    # --- Retrieved context tab ---
    with tab_retrieval:
        if not st.session_state.last_retrieved:
            st.info("Ask a question in the Chat tab to see retrieved chunks here.")
        else:
            for r in st.session_state.last_retrieved:
                st.markdown(
                    f"<div class='chunk-card'><div class='chunk-meta'>{r['source']} · chunk {r['index']} · score {r['score']:.3f}</div>{r['text']}</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# ---------------------------------  RUN  --------------------------------------
# ---------------------------------------------------------------------------
init_state()

if not st.session_state.logged_in:
    inject_theme(st.session_state.theme)
    login_screen()
else:
    main_app()