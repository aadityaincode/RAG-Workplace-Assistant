# RAG Workplace Assistant

A minimal Retrieval-Augmented Generation (RAG) system for querying workplace documents using ChromaDB and Google's Gemini AI.

## Features

- Semantic search across workplace documents
- AI-powered responses using Gemini 2.5 Flash
- Source document retrieval and display
- Clean, dark minimal UI
- Built with Flask, ChromaDB, and Sentence Transformers

---

## Prerequisites

- Python 3.11 or higher
- Google Gemini API key — [Get one here](https://aistudio.google.com/app/apikey)

---

## Project Structure

Place all documents you want to search as plain `.txt` files inside the `documents/` folder:

```
documents/
├── 00_Project_Overview.txt
├── 01_Business_Context.txt
├── 02_System_Architecture.txt
├── 03_API_Documentation.txt
├── 04_Database_Schema.txt
├── 05_IT_Security_Guidelines.txt
├── 06_Deployment_Guide.txt
├── 07_Troubleshooting_Guide.txt
├── 08_Internal_FAQ.txt
├── 09_Meeting_Notes.txt
├── 10_Employee_Handbook_Excerpt.txt
├── 11_Incident_Report_2025-11-04.txt
└── 12_Random_Internal_Notes.txt
```

You may add your own `.txt` files freely. Subfolders are supported.

---

## Setup

### Quick Start (Recommended)

**macOS/Linux:**
```bash
./script.sh
```

**Windows:**
```bat
script.bat
```

This automatically creates a virtual environment and installs all dependencies.

### Manual Setup

**1. Create a virtual environment**
```bash
python3.11 -m venv venv
```

**2. Activate the virtual environment**

macOS/Linux:
```bash
source venv/bin/activate
```
Windows:
```bat
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your API key**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_api_key_here
```

> If the `.env` file is missing or the key is not set, the app will raise:
> `ValueError: GOOGLE_API_KEY not found. Please set it in your .env file.`

---

## Ingesting Documents

After adding or updating files in the `documents/` folder, run the ingestion script to populate the vector database:

```bash
python ingest.py
```

This processes all `.txt` files and stores their embeddings in ChromaDB. Re-run this any time your documents change.

---

## Running the App

This project includes two interfaces: a **Flask app** (recommended for local use) and a **Streamlit app** (for Google Colab or remote access). Both connect to the same ChromaDB backend and produce identical results.

### Option 1: Flask — Recommended for Local Use

The Flask app is the primary interface for running the assistant locally after cloning the repo.

```bash
python app.py
```

Then open your browser and go to:
```
http://localhost:5000
```

### Option 2: Streamlit — For Google Colab or Remote Access

The Streamlit app is included specifically for environments where Flask is not practical, such as **Google Colab**. If you are running this locally after cloning the repo, use the Flask app above.

```bash
streamlit run streamlit_app.py
```

Then open your browser and go to the URL shown in the terminal (usually `http://localhost:8501`).

> **Running on Google Colab?** The Streamlit app requires ngrok to expose the app publicly from Colab's isolated environment. See [Colab_Setup_Guide.md](./Colab_Setup_Guide.md) for full step-by-step instructions.

---

## Usage

1. Type your question in the search box (e.g., *What is the business context?*)
2. Click **Search** or press **Enter**
3. The AI-generated response appears on the left
4. Source documents used to generate the answer appear on the right

---

## Troubleshooting

**`ValueError: GOOGLE_API_KEY not found`**
Your `.env` file is missing or the variable name is misspelled. Ensure the file exists in the project root and contains exactly `GOOGLE_API_KEY=your_key_here`.

**Empty or irrelevant responses**
Your documents may not have been ingested yet, or the ChromaDB database is stale. Re-run `python ingest.py` after any changes to the `documents/` folder.

**ChromaDB errors on startup**
The database may be corrupted. Delete the `chroma_db/` directory and re-run ingestion:
```bash
rm -rf chroma_db
python ingest.py
```