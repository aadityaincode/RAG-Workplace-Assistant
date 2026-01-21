import os
import subprocess
import google.genai as genai
from flask import Flask, request, jsonify, render_template
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAsjrEu5-nJKDNHEr72F_icskuGcfjChN4")
if API_KEY == "YOUR_API_KEY":
    print("Warning: Using a placeholder API key. Please set the GOOGLE_API_KEY environment variable.")

# Initialize the client with the new API
client_genai = genai.Client(api_key=API_KEY)

app = Flask(__name__)

# --- ChromaDB Client and Model ---
client = chromadb.Client()
collection = None
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_database():
    """Calls the ingestion script to populate ChromaDB."""
    global collection
    print("Initializing database...")
    try:
        # Import and run the ingest function directly instead of subprocess
        from ingest import ingest_documents
        ingest_documents()
        collection = client.get_collection("workplace_documents")
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Try to get existing collection, or create empty one
        try:
            collection = client.get_collection("workplace_documents")
        except:
            collection = client.create_collection("workplace_documents")

def generate_response(context, query):
    """Generates a response using the generative model."""
    # Build the documents string separately
    docs_text = "\n".join(f"- {doc}" for doc in context)
    
    prompt = f"""
    You are a helpful workplace assistant. Based on the following documents, please answer the user's question.

    Documents:
    {docs_text}

    Question: {query}

    Answer:
    """
    try:
        response = client_genai.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Performs RAG to answer a user's query."""
    if collection is None:
        return jsonify({"error": "Database not initialized"}), 500
        
    data = request.json
    query_text = data.get('query')
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    # 1. Retrieval
    query_embedding = embedding_model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    retrieved_docs = results['documents'][0]

    # 2. Augmentation & 3. Generation
    generated_answer = generate_response(retrieved_docs, query_text)

    return jsonify({
        "generated_answer": generated_answer,
        "retrieved_documents": retrieved_docs
    })

if __name__ == '__main__':
    initialize_database()
    app.run(port=5000, debug=True)
