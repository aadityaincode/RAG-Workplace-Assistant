import os
import chromadb
from sentence_transformers import SentenceTransformer

def ingest_documents():
    """
    Reads documents from the 'documents' directory, generates embeddings,
    and stores them in a ChromaDB collection.
    """
    client = chromadb.Client()
    # It's good practice to delete the collection if it exists to ensure a fresh start
    if "workplace_documents" in [c.name for c in client.list_collections()]:
        client.delete_collection("workplace_documents")
        
    collection = client.create_collection("workplace_documents")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    doc_dir = "documents"  # Changed this line
    documents = []
    doc_ids = []
    
    # In a real RAG system, you'd have a more sophisticated chunking strategy.
    # For this demo, we'll treat each file as a single document/chunk.
    for i, filename in enumerate(os.listdir(doc_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir, filename), 'r') as f:
                text = f.read()
                documents.append(text)
                doc_ids.append(str(i))

    if not documents:
        print("No documents found to ingest.")
        return

    # Generate embeddings and add to the collection
    embeddings = model.encode(documents).tolist()
    collection.add(
        embeddings=embeddings,
        documents=documents,
        ids=doc_ids
    )

    print(f"Successfully ingested {len(documents)} documents into ChromaDB.")

if __name__ == '__main__':
    ingest_documents()