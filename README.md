# RAG Workplace Assistant 🤖

A minimal RAG (Retrieval-Augmented Generation) system for querying workplace documents using ChromaDB and Google's Gemini AI.

## Features

- Semantic search across workplace documents
- AI-powered responses using Gemini 2.5 Flash
- Source document retrieval and display
- Clean, dark minimal UI
- Built with Flask, ChromaDB, and Sentence Transformers

## Prerequisites

- Python 3.11+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

## Setup Instructions

### 1. Create a Virtual Environment
```bash
python3.11 -m venv venv
```

### 2. Activate the Virtual Environment
**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root directory:
```bash
GOOGLE_API_KEY=your_api_key_here
```

**⚠️ Important:** If you don't set up the `.env` file, you'll get this error:
```
ValueError: GOOGLE_API_KEY not found. Please set it in your .env file.
```

### 5. Add Your Documents

Place your `.txt` documents in the `documents/` folder. The system will automatically ingest them on startup.

### 6. Run the Application
```bash
python app.py
```

You should see output like:
```
Initializing database...
Successfully ingested 3 documents into ChromaDB.
Database initialized successfully.
 * Running on http://127.0.0.1:5000
```

### 7. Open in Browser

Copy the link from the terminal and paste it into your browser:
```
http://127.0.0.1:5000
```

Or simply navigate to: **http://localhost:5000**

## Usage

1. Type your question in the search box (e.g., "What is Project Titan?")
2. Click **Search** or press **Enter**
3. View the AI-generated response on the left
4. See the source documents on the right

