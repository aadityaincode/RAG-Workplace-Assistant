import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# --- Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

app = Flask(__name__)

# --- ChromaDB Client and Embedding Model ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = None

embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')


def initialize_database():
    global collection

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


def generate_response(context, query):
    docs_text = "\n".join(f"- {doc}" for doc in context)

    prompt = f"""You are a helpful workplace assistant. Answer the question based ONLY on the provided documents.

Context Documents:
{docs_text}

Question: {query}

Instructions:
- If the answer is not in the documents, say "I don't have enough information to answer that."
- Be concise and accurate
- Use a professional but friendly tone

Answer:"""

    response = model.generate_content(prompt)
    return response.text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    if collection is None:
        return jsonify({"error": "Database not initialized"}), 500

    data = request.json
    query_text = data.get('query')

    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    query_embedding = embedding_model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieved_docs = results['documents'][0]

    if not retrieved_docs or all(not doc.strip() for doc in retrieved_docs):
        return jsonify({
            "generated_answer": "I don't have enough information to answer that.",
            "retrieved_documents": []
        })

    top_docs = retrieved_docs[:3]
    generated_answer = generate_response(top_docs, query_text)

    return jsonify({
        "generated_answer": generated_answer,
        "retrieved_documents": top_docs
    })


if __name__ == '__main__':
    initialize_database()
    app.run(port=5000, debug=True)