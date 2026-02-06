"""
Document Ingestion Module for RAG System

This module handles the ingestion pipeline for processing workplace documents
into a vector database. It reads text files, splits them into chunks, generates
embeddings, and stores them in ChromaDB for semantic search.

Author: Aaditya Dhungana & Emmanuel Buhari
Date: February 2026
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ingest_documents():
    """
    Ingests documents from the /documents directory into ChromaDB.
    
    This function implements a complete RAG ingestion pipeline:
    1. Reads all .txt files from the documents directory
    2. Splits large documents into smaller chunks with overlap
    3. Generates vector embeddings for each chunk
    4. Stores chunks and embeddings in a persistent ChromaDB collection
    
    The chunking strategy uses RecursiveCharacterTextSplitter to maintain
    semantic coherence while creating manageable chunk sizes.
    
    Returns:
        None. Prints status messages to console.
    """
    
    # Initialize ChromaDB with persistent storage
    # PersistentClient saves data to disk, unlike the ephemeral Client()
    # This ensures data survives application restarts
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Check if collection exists and delete it for a clean start
    # This is useful during development but may be removed in production
    collection_name = "workplace_documents"
    existing_collections = [c.name for c in client.list_collections()]
    
    if collection_name in existing_collections:
        client.delete_collection(collection_name)
        print(f"Deleted existing '{collection_name}' collection for fresh ingestion.")
    
    # Create a new collection to store document embeddings
    # Collections in ChromaDB are like tables in a traditional database
    collection = client.create_collection(collection_name)
    
    # Initialize the embedding model
    # BAAI/bge-small-en-v1.5 is a state-of-the-art sentence embedding model
    # - Produces 384-dimensional vectors
    # - Better performance than older models like all-MiniLM-L6-v2
    # - Optimized for semantic search tasks
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Configure text chunking strategy
    # RecursiveCharacterTextSplitter splits text intelligently:
    # - chunk_size: Target size for each chunk (~500 chars = ~100 words)
    # - chunk_overlap: Overlap between chunks to preserve context
    # - separators: Priority list for where to split (paragraphs > sentences > words)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,           # Approximate characters per chunk
        chunk_overlap=30,         # Characters of overlap between chunks
        separators=["\n\n", "\n", ". ", " ", ""]  # Split priority
    )
    
    # Define the directory containing source documents
    doc_dir = "documents"
    
    # Validate that the documents directory exists
    if not os.path.exists(doc_dir):
        print(f"ERROR: Directory '{doc_dir}' does not exist!")
        print(f"Please create the directory and add .txt files to it.")
        return
    
    # Initialize storage for processed chunks
    all_chunks = []        # Stores the actual text chunks
    all_metadatas = []     # Stores metadata for each chunk (source file, position, etc.)
    chunk_ids = []         # Unique identifiers for each chunk
    
    chunk_counter = 0      # Global counter for unique IDs
    file_count = 0         # Number of files processed
    
    # Process each file in the documents directory
    print(f"Processing documents from '{doc_dir}'...")
    for filename in os.listdir(doc_dir):
        # Only process .txt files, skip other file types
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(doc_dir, filename)
        file_count += 1
        
        # Read the entire file content
        # Using utf-8 encoding to handle special characters
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"WARNING: Could not read {filename}: {e}")
            continue
        
        # Split the document into chunks using the configured strategy
        chunks = text_splitter.split_text(text)
        print(f"  {filename}: {len(chunks)} chunks created")
        
        # Store each chunk with its metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            
            # Metadata helps track source and context for each chunk
            # Useful for debugging and displaying sources to users
            all_metadatas.append({
                "source": filename,           # Original filename
                "chunk_index": i,             # Position within the source document
                "total_chunks": len(chunks)   # Total chunks from this document
            })
            
            # Create a unique ID for this chunk
            # Format: filename_globalcounter (e.g., "policy.txt_0")
            chunk_ids.append(f"{filename.replace('.txt', '')}_{chunk_counter}")
            chunk_counter += 1
    
    # Validation: Ensure we found and processed documents
    if not all_chunks:
        print("WARNING: No .txt documents found in the 'documents' folder.")
        print("Please add .txt files and run the ingestion again.")
        return
    
    print(f"\nProcessed {file_count} files into {len(all_chunks)} total chunks.")
    
    # Generate embeddings for all chunks
    # This converts text into numerical vectors for semantic search
    # The embedding model encodes semantic meaning, allowing similarity search
    print("Generating embeddings (this may take a moment)...")
    embeddings = embedding_model.encode(
        all_chunks, 
        show_progress_bar=True,    # Display progress during encoding
    ).tolist()
    
    # Store everything in the ChromaDB collection
    # ChromaDB will index these embeddings for fast similarity search
    print("Storing in ChromaDB...")
    collection.add(
        embeddings=embeddings,      # Vector representations for semantic search
        documents=all_chunks,       # Original text for display to users
        metadatas=all_metadatas,    # Source tracking and debugging info
        ids=chunk_ids               # Unique identifiers (required by ChromaDB)
    )
    
    print(f"\nSUCCESS: Ingested {len(all_chunks)} chunks from {file_count} documents.")
    print(f"Data stored in: ./chroma_db")


# Allow this script to be run standalone for testing
if __name__ == '__main__':
    print("=" * 60)
    print("RAG Document Ingestion Script")
    print("=" * 60)
    ingest_documents()
    print("=" * 60)