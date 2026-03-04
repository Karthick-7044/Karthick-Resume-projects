import streamlit as st
from google import genai
from google.genai import types
import json, time, io
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📓 NotebookLM – Industry RAG",
    page_icon="📓",
    layout="wide",
)

# ─────────────────────────────────────────────
# WHITE NOTEBOOK STYLE  ← UNTOUCHED
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #1a73e8;
    --bg: #ffffff;
    --surface: #f8f9fa;
    --text: #202124;
    --muted: #5f6368;
    --border: #e0e0e0;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

.card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid var(--border);
    margin-bottom: 1rem;
}

.chat-user { display: flex; justify-content: flex-end; margin: 0.5rem 0; }
.chat-user .bubble {
    background: var(--primary);
    color: white;
    padding: 0.7rem 1rem;
    border-radius: 14px 14px 4px 14px;
    max-width: 75%;
}

.chat-ai { display: flex; justify-content: flex-start; margin: 0.5rem 0; }
.chat-ai .bubble {
    background: #f1f3f4;
    border: 1px solid var(--border);
    padding: 0.7rem 1rem;
    border-radius: 14px 14px 14px 4px;
    max-width: 80%;
}

.chat-label { font-size: 0.7rem; color: var(--muted); margin-bottom: 0.2rem; }

.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 6px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODELS  ← UNTOUCHED
# ─────────────────────────────────────────────
MODELS = {
    "🤖 Auto (try all models)": "auto",

    # Premium / High Quality
    "🧠 Gemini 2.5 Pro (Best Quality)": "gemini-2.5-pro",
    "🚀 Gemini Pro Latest": "gemini-pro-latest",
    "🧪 Gemini Experimental 1206": "gemini-exp-1206",

    # Flash / Fast
    "⚡ Gemini 2.5 Flash": "gemini-2.5-flash-preview-05-20",
    "🔥 Gemini Flash Latest": "gemini-flash-latest",

    # Free Tier Optimized
    "🪶 Gemini 2.5 Flash-Lite (Most Quota)": "gemini-2.5-flash-lite-preview-06-17",
    "🧠 Gemini 2.0 Flash (Stable)": "gemini-2.0-flash",
}

FALLBACK_ORDER = [
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemini-exp-1206",
    "gemini-2.5-flash-preview-05-20",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.0-flash",
]

# MIME type map for FileSearch API
MIME_MAP = {
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv":  "text/csv",
    ".md":   "text/plain",
}


# ─────────────────────────────────────────────
# FILESEARCH API HELPERS  ← replaces extract_text + context dump
# ─────────────────────────────────────────────

def get_client(api_key):
    """Return a google.genai Client (new SDK, required for FileSearch)."""
    return genai.Client(api_key=api_key)


def get_or_create_store(client, store_name="NotebookLM-Store"):
    """Reuse existing store by display_name, or create a new one."""
    for store in client.file_search_stores.list():
        if store.display_name == store_name:
            return store
    return client.file_search_stores.create(
        config={"display_name": store_name}
    )


def upload_to_store(client, store, uploaded_file):
    """
    Upload a Streamlit UploadedFile directly to FileSearchStore.
    Google handles chunking + embedding automatically.
    Returns (success: bool, message: str)
    """
    name = uploaded_file.name
    ext  = "." + name.rsplit(".", 1)[-1].lower()
    mime = MIME_MAP.get(ext, "application/octet-stream")

    raw_bytes = uploaded_file.read()
    buf = io.BytesIO(raw_bytes)
    buf.name = name          # SDK needs .name on the stream
    buf.seek(0)

    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=buf,
            file_search_store_name=store.name,
            config={
                "display_name": name,
                "mime_type": mime,
            },
        )

        # Poll until indexing is done
        max_wait, waited = 120, 0
        while not operation.done:
            time.sleep(4)
            waited += 4
            operation = client.operations.get(operation)
            if waited >= max_wait:
                return False, f"Indexing timeout after {max_wait}s"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def chat_with_filesearch(model_choice, system_prompt, history, user_msg, store_name, client):
    """
    Query Gemini with FileSearch tool — Google does vector search, returns grounded answer.
    Falls back through model list on 404/429.
    """
    models = FALLBACK_ORDER if model_choice == "auto" else [model_choice]

    sdk_history = []
    for h in history:
        role    = "user" if h["role"] == "user" else "model"
        sdk_history.append({"role": role, "parts": [{"text": h["content"]}]})

    last_error = ""
    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=sdk_history + [{"role": "user", "parts": [{"text": user_msg}]}],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_name]
                            )
                        )
                    ],
                ),
            )
            return response.text, model_name

        except Exception as e:
            err = str(e)
            last_error = err
            if any(x in err for x in ["404", "not found", "429", "quota", "rate"]):
                continue
            return f"❌ Error: {err}", None

    return f"❌ All models failed. Last error: {last_error}", None


def build_system_prompt(industry):
    return (
        f"You are an expert {industry} Knowledge Assistant similar to NotebookLM. "
        "You must answer using ONLY the provided documents. "
        "If not found, respond exactly with: "
        "'📭 This information is not available in the uploaded knowledge base.' "
        "Always mention which document you sourced from."
    )


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for k, v in {
    "chat_history": [],
    "uploaded_files": [],   # list of filenames indexed in FileSearch store
    "gemini_configured": False,
    "client": None,
    "store": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# HEADER  ← UNTOUCHED
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0;">
<h1 style="color:#1a73e8;">📓 NotebookLM</h1>
<p style="color:#5f6368;">Industry Knowledge Base · Multi-Model Gemini · FileSearch RAG</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("Gemini API Key", type="password")
    if api_key and not st.session_state.gemini_configured:
        try:
            client = get_client(api_key)
            store  = get_or_create_store(client, "NotebookLM-Store")
            st.session_state.client = client
            st.session_state.store  = store
            st.session_state.gemini_configured = True
            st.success(f"API Connected · Store ready")
            st.caption(f"`{store.name}`")
        except Exception as e:
            st.error(str(e))

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model_name  = MODELS[model_label]

    industry = st.text_input("Industry", value="Medical")

    st.subheader("Upload Documents")
    st.caption("PDF, DOCX, TXT, XLSX, CSV — Google indexes & chunks automatically")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx", "xlsx", "csv", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.session_state.gemini_configured:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_files:
                with st.spinner(f"Indexing {f.name}…"):
                    ok, msg = upload_to_store(
                        st.session_state.client,
                        st.session_state.store,
                        f,
                    )
                if ok:
                    st.session_state.uploaded_files.append(f.name)
                    st.success(f"✅ {f.name} indexed")
                else:
                    st.error(f"❌ {f.name}: {msg}")
    elif uploaded_files and not st.session_state.gemini_configured:
        st.warning("Enter API key first to index documents.")

    if st.session_state.uploaded_files:
        st.success(f"{len(st.session_state.uploaded_files)} file(s) in store")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────
# MAIN CHAT  ← UNTOUCHED layout
# ─────────────────────────────────────────────
col_chat, col_docs = st.columns([3, 1])

with col_docs:
    st.markdown('<div class="card"><b>Documents in Store</b></div>', unsafe_allow_html=True)
    if st.session_state.uploaded_files:
        for fname in st.session_state.uploaded_files:
            st.write("📄", fname)
    else:
        st.write("No documents indexed yet.")

    if st.session_state.store:
        st.caption(f"Store: `{st.session_state.store.name}`")

with col_chat:

    if not st.session_state.chat_history:
        st.info("Upload documents and start asking questions.")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div>
                    <div class="chat-label">You</div>
                    <div class="bubble">{msg["content"]}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-ai">
                <div>
                    <div class="chat-label">Notebook</div>
                    <div class="bubble">{msg["content"]}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask your notebook...", height=80)
        submitted  = st.form_submit_button("Ask")

    if submitted and user_input.strip():
        if not st.session_state.gemini_configured:
            st.error("Enter Gemini API key.")
        elif not st.session_state.uploaded_files:
            st.error("Upload at least one document first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner("Searching documents…"):
                reply, used_model = chat_with_filesearch(
                    model_name,
                    build_system_prompt(industry),
                    st.session_state.chat_history[:-1],
                    user_input,
                    st.session_state.store.name,
                    st.session_state.client,
                )

            if used_model:
                reply = f"{reply}\n\n---\n_Model used: {used_model}_"

            st.session_state.chat_history.append({"role": "model", "content": reply})
            st.rerun()


# ─────────────────────────────────────────────
# EXPORT  ← UNTOUCHED
# ─────────────────────────────────────────────
if st.session_state.chat_history:
    export = json.dumps(st.session_state.chat_history, indent=2)
    st.download_button(
        "Download Chat JSON",
        export,
        file_name=f"notebooklm_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )
