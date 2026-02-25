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
    # This ensures that re-running ingestion always reflects the latest documents
    collection_name = "workplace_documents"
    existing_collections = [c.name for c in client.list_collections()]

    if collection_name in existing_collections:
        client.delete_collection(collection_name)
        print(f"Deleted existing '{collection_name}' collection for fresh ingestion.")

    # Create a new collection to store document embeddings
    collection = client.create_collection(collection_name)

    # Initialize the sentence embedding model
    # BAAI/bge-small-en-v1.5 is a high-quality, lightweight embedding model
    # - Produces 384-dimensional vectors
    # - Strong performance on semantic search benchmarks
    # - Faster and smaller than larger models while retaining quality
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    # Configure the text chunking strategy
    #
    # chunk_size=800: Each chunk is approximately 800 characters (~150 words),
    # large enough to contain a full policy section or several related sentences.
    # Smaller chunks (e.g. 200) lose surrounding context, causing the model to
    # answer with "I don't have enough information" even when the answer exists.
    #
    # chunk_overlap=100: 100 characters of overlap between adjacent chunks ensures
    # that sentences sitting at a chunk boundary appear in both chunks, preventing
    # information loss at split points.
    #
    # separators: Split priority goes from largest natural boundary (paragraph)
    # down to smallest (character), preserving semantic units where possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Define the directory containing source documents
    doc_dir = "documents"

    # Validate that the documents directory exists before proceeding
    if not os.path.exists(doc_dir):
        print(f"ERROR: Directory '{doc_dir}' does not exist!")
        print("Please create the directory and add .txt files to it.")
        return

    # Initialize storage for processed chunks and their metadata
    all_chunks = []      # The actual text content of each chunk
    all_metadatas = []   # Metadata for each chunk (source file, position)
    chunk_ids = []       # Unique identifiers required by ChromaDB

    chunk_counter = 0    # Global counter used to generate unique IDs
    file_count = 0       # Tracks how many files were successfully processed

    # Process each file in the documents directory
    print(f"Processing documents from '{doc_dir}'...")
    for filename in sorted(os.listdir(doc_dir)):
        # Only process plain text files
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(doc_dir, filename)
        file_count += 1

        # Read the file contents, handling encoding gracefully
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"WARNING: Could not read {filename}: {e}")
            continue

        # Split the document into overlapping chunks
        chunks = text_splitter.split_text(text)
        print(f"  {filename}: {len(chunks)} chunks")

        # Record each chunk alongside its source metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            # Metadata is stored alongside each chunk in ChromaDB.
            # It is returned with search results and used to show users
            # which document their answer came from.
            all_metadatas.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

            # Unique ID format: <filename_without_extension>_<global_counter>
            chunk_ids.append(f"{filename.replace('.txt', '')}_{chunk_counter}")
            chunk_counter += 1

    # Stop early if no documents were found
    if not all_chunks:
        print("WARNING: No .txt documents found in the 'documents' folder.")
        print("Please add .txt files and run ingestion again.")
        return

    print(f"\nProcessed {file_count} files into {len(all_chunks)} total chunks.")

    # Generate vector embeddings for all chunks in a single batch call
    # Embeddings convert text into numerical vectors that capture semantic meaning,
    # enabling similarity search (finding chunks that mean the same thing as the query)
    print("Generating embeddings (this may take a moment)...")
    embeddings = embedding_model.encode(
        all_chunks,
        show_progress_bar=True
    ).tolist()

    # Store all chunks, embeddings, and metadata in ChromaDB
    # ChromaDB indexes the embeddings for fast approximate nearest-neighbour search
    print("Storing in ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=chunk_ids
    )

    print(f"\nSUCCESS: Ingested {len(all_chunks)} chunks from {file_count} documents.")
    print("Data stored in: ./chroma_db")


# Allow this script to be run standalone to (re)populate the database
if __name__ == '__main__':
    print("=" * 60)
    print("RAG Document Ingestion Script")
    print("=" * 60)
    ingest_documents()
    print("=" * 60)