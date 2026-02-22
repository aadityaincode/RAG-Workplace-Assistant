# RAG Workplace Assistant

A minimal Retrieval-Augmented Generation (RAG) system for querying workplace documents using ChromaDB and Google's Gemini AI.

## Features

- Semantic search across workplace documents
- AI-powered responses using Gemini 2.5 Flash
- Source document retrieval and display
- Clean, dark minimal UI
- Built with Flask, ChromaDB, and Sentence Transformers

## Prerequisites

- Python 3.11 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

## Project Structure

All documents for search and retrieval should be placed as plain text (.txt) files in the `documents/` folder. Example structure:

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

You may add your own .txt files.

## Setup Instructions

### Quick Start (Recommended)

#### macOS/Linux:
Run the setup script:
```bash
./script.sh
```

#### Windows:
Run the batch script:
```bat
script.bat
```

This will create a virtual environment and install all dependencies.

### Manual Setup

1. **Create a Virtual Environment**
	```bash
	python3.11 -m venv venv
	```
2. **Activate the Virtual Environment**
	- macOS/Linux:
		```bash
		source venv/bin/activate
		```
	- Windows:
		```bat
		.\venv\Scripts\activate
		```
3. **Install Dependencies**
	```bash
	pip install -r requirements.txt
	```
4. **Configure API Key**
	- Create a `.env` file in the project root directory with:
		```
		GOOGLE_API_KEY=your_api_key_here
		```
	- If you don't set up the `.env` file, you'll get:
		```
		ValueError: GOOGLE_API_KEY not found. Please set it in your .env file.
		```

## Ingesting Documents

After adding or updating files in the `documents/` folder, run:

```bash
python ingest.py
```

This will process all .txt files (including subfolders) and update the ChromaDB vector database.

If you want to reset the database, run:

```bash
./reset_chromadb.sh
```

Then re-run `python ingest.py`.

## Running the Application

Start the Flask app:

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```
or
```
http://localhost:5000
```

## Usage

1. Type your question in the search box (e.g., What is the business context?)
2. Click Search or press Enter
3. View the AI-generated response on the left
4. See the source documents on the right

## Testing

You can add and run tests in `test.py` to check API endpoints, ingestion, search, and error handling.

---
**Note:** All document files must be plain text (.txt) with no markdown formatting for best results.
