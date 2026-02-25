# Standard library and third-party imports
import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# --- Configuration ---
# Retrieve Gemini API key from environment
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure Gemini API and load the generative model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Initialize Flask web application
app = Flask(__name__)

# --- ChromaDB Client and Embedding Model ---
# Create persistent ChromaDB client for vector database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Placeholder for ChromaDB collection, will be set after initialization
collection = None

# Load sentence embedding model for semantic search
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Relevance threshold for filtering out low-confidence ChromaDB results.
RELEVANCE_THRESHOLD = 1.0


# Ensure ChromaDB collection exists and is populated
def initialize_database():
    global collection

    try:
        # Try to get the existing collection
        collection = chroma_client.get_collection("workplace_documents")
        # If collection exists but is empty, run ingestion
        if collection.count() == 0:
            from ingest import ingest_documents
            ingest_documents()
            collection = chroma_client.get_collection("workplace_documents")
    except Exception as e:
        # If collection does not exist, run ingestion
        print(f"Collection not found, running ingestion: {e}")
        from ingest import ingest_documents
        ingest_documents()
        collection = chroma_client.get_collection("workplace_documents")


# Generate an AI response using Gemini, grounded in context documents
def generate_response(context, query):
    # Combine context documents into a single string
    docs_text = "\n".join(f"- {doc}" for doc in context)

    # Build prompt for Gemini model
    prompt = f"""You are a helpful workplace assistant for Arrowood & Partners.
Answer the employee's question using the context documents below.

Context Documents:
{docs_text}

Question: {query}

Instructions:
- Answer directly and confidently if the information is present, even partially
- Do not say "I don't have enough information" if any relevant detail exists in the documents
- Only say "I don't have enough information to answer that" if the topic is completely absent from the documents
- Be concise and use a professional but friendly tone

Answer:"""

    # Generate answer using Gemini model
    response = model.generate_content(prompt)
    return response.text


# Home page route
@app.route('/')
def index():
    return render_template('index.html')


# Search endpoint: handles user queries and returns AI answers
@app.route('/search', methods=['POST'])
def search():
    # Ensure database is initialized
    if collection is None:
        return jsonify({"error": "Database not initialized"}), 500

    # Parse incoming JSON request
    data = request.json
    query_text = data.get('query')

    # Validate query input
    if not query_text:
        return jsonify({"error": "Query text is required"}), 400

    # Embed the query for semantic search
    query_embedding = embedding_model.encode(query_text).tolist()

    # Query ChromaDB for top matching documents, including distance scores
    # for relevance filtering
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "distances"]
    )

    # Retrieve documents and their similarity distances from results
    retrieved_docs = results['documents'][0]
    distances = results['distances'][0]

    # Filter out chunks that are too dissimilar to the query.
    # This prevents unrelated document chunks from being passed to Gemini,
    # which causes confident-sounding but incorrect answers (e.g. responding
    # to "hello" with onboarding content that happened to match semantically).
    filtered_docs = [
        doc for doc, dist in zip(retrieved_docs, distances)
        if dist < RELEVANCE_THRESHOLD
    ]

    # If no chunks passed the relevance threshold, return fallback answer
    if not filtered_docs:
        return jsonify({
            "generated_answer": "I don't have enough information to answer that.",
            "retrieved_documents": []
        })

    # Generate answer using all filtered docs for maximum context
    generated_answer = generate_response(filtered_docs, query_text)

    # Return answer and source documents to frontend
    return jsonify({
        "generated_answer": generated_answer,
        "retrieved_documents": filtered_docs
    })


# Entry point: initialize database and start Flask server
if __name__ == '__main__':
    initialize_database()
    app.run(port=5000, debug=True)