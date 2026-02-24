import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="Workplace Assistant",
    layout="wide"
)

# --- Load environment variables ---
# --- Load Gemini API key: try Colab secrets, then .env/environment variable ---
API_KEY = None
try:
    from google.colab import userdata
    API_KEY = userdata.get('GOOGLE_API_KEY')
except Exception:
    pass

if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please add it as a Colab secret, or create a .env file in the project root with GOOGLE_API_KEY=your_key_here, or set it as an environment variable.")
    st.stop()

# --- Configure Gemini ---
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- ChromaDB client and embedding model ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# --- Database initialization ---
def initialize_database():
    try:
        collection = chroma_client.get_collection("workplace_documents")
        if collection.count() == 0:
            from ingest import ingest_documents
            ingest_documents()
            collection = chroma_client.get_collection("workplace_documents")
    except:
        from ingest import ingest_documents
        ingest_documents()
        collection = chroma_client.get_collection("workplace_documents")
    return chroma_client.get_collection("workplace_documents")

# --- Generate AI response ---
def generate_response(context, query):
    docs_text = "\n".join(f"- {doc}" for doc in context)
    prompt = f"""You are a helpful workplace assistant. Answer the question based ONLY on the provided documents.\n\nContext Documents:\n{docs_text}\n\nQuestion: {query}\n\nInstructions:\n- If the answer is not in the documents, say "I don't have enough information to answer that.\"\n- Be concise and accurate\n- Use a professional but friendly tone\n\nAnswer:"""
    response = model.generate_content(prompt)
    return response.text

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0f172a;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Page title */
    .page-title {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .page-title h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.3rem;
    }
    .page-title p {
        font-size: 1.1rem;
        color: #94a3b8;
    }

    /* Search input */
    .stTextInput > div > div > input {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
        border: 2px solid #334155 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
    }
    .stTextInput label { display: none; }

    /* Search button */
    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100%;
        height: 42px;
        margin-top: 1px;
    }
    .stButton > button:hover {
        background-color: #2563eb !important;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4) !important;
    }

    /* Search wrapper - style via Streamlit's horizontal block */
    [data-testid="stHorizontalBlock"]:first-of-type {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }

    /* Panel cards */
    .panel-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        min-height: 400px;
    }
    .panel-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #334155;
    }
    .response-text {
        color: #e2e8f0;
        line-height: 1.8;
        font-size: 1rem;
    }
    .placeholder-text {
        color: #64748b;
        text-align: center;
        padding: 3rem 1rem;
        font-style: italic;
        font-size: 0.95rem;
    }
    .source-card {
        background: #334155;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-radius: 6px;
    }
    .source-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .source-text {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.6;
    }
    .error-text {
        color: #f87171;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize collection in session state (silent, no spinner shown to user) ---
if "collection" not in st.session_state:
    st.session_state.collection = initialize_database()

# --- Initialize result state ---
if "response_html" not in st.session_state:
    st.session_state.response_html = "<div class='placeholder-text'>Your answer will appear here. Try asking about vacation policies, project details, or IT guidelines!</div>"
if "sources_html" not in st.session_state:
    st.session_state.sources_html = "<div class='placeholder-text'>Retrieved documents will appear here</div>"

# --- Header ---
st.markdown("""
<div class="page-title">
    <h1>Workplace Assistant</h1>
    <p>Ask questions about your workplace documents and policies</p>
</div>
""", unsafe_allow_html=True)

# --- Search Bar ---
search_col, btn_col = st.columns([6, 1])
with search_col:
    query_text = st.text_input(
        "query",
        placeholder="Ask me anything about workplace policies, projects, or guidelines...",
        label_visibility="collapsed",
        key="queryText"
    )
with btn_col:
    search_clicked = st.button("Search", key="searchBtn", use_container_width=True)

# --- Run search logic BEFORE rendering panels so state is ready ---
if search_clicked:
    if not query_text:
        st.session_state.response_html = "<div class='error-text'>Please enter a question.</div>"
        st.session_state.sources_html = "<div class='placeholder-text'>Retrieved documents will appear here</div>"
    else:
        collection = st.session_state.collection
        if collection is None:
            st.session_state.response_html = "<div class='error-text'>Database not initialized.</div>"
            st.session_state.sources_html = "<div class='placeholder-text'>Retrieved documents will appear here</div>"
        else:
            with st.spinner("Thinking..."):
                try:
                    query_embedding = embedding_model.encode(query_text).tolist()
                    results = collection.query(query_embeddings=[query_embedding], n_results=5)
                    retrieved_docs = results['documents'][0]

                    if not retrieved_docs or all(not doc.strip() for doc in retrieved_docs):
                        st.session_state.response_html = "<div class='response-text'>I don't have enough information to answer that.</div>"
                        st.session_state.sources_html = "<div class='placeholder-text'>No sources found</div>"
                    else:
                        top_docs = retrieved_docs[:3]
                        generated_answer = generate_response(top_docs, query_text)

                        st.session_state.response_html = f"<div class='response-text'>{generated_answer}</div>"
                        st.session_state.sources_html = "".join([
                            f"<div class='source-card'><div class='source-label'>Source {i+1}</div><div class='source-text'>{doc}</div></div>"
                            for i, doc in enumerate(top_docs)
                        ])
                except Exception as e:
                    st.session_state.response_html = f"<div class='error-text'>Error: {str(e)}</div>"
                    st.session_state.sources_html = "<div class='placeholder-text'>Unable to retrieve sources</div>"

# --- Result panels rendered AFTER state is fully updated ---
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-header">Response</div>
        {st.session_state.response_html}
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-header">Source Documents</div>
        {st.session_state.sources_html}
    </div>
    """, unsafe_allow_html=True)