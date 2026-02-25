# Running the RAG Workplace Assistant on Google Colab

Since Colab doesn't support persistent local servers the way a traditional environment does, this guide walks you through tunneling the Streamlit app to a public URL using **ngrok**, and authenticating with the **Gemini API** via Google AI Studio.

---

## Prerequisites

Before running any cells, make sure you have accounts set up for:
- [Google AI Studio](https://aistudio.google.com) — for your Gemini API key
- [ngrok](https://dashboard.ngrok.com/signup) — for exposing the Streamlit app publicly

Both are free to sign up for. Steps to get the required tokens are covered in **Step 3** below.

---

## Step 1: Clone and Prepare the Project

Clone the repository from GitHub and navigate into the project directory. All subsequent commands must be run from within this folder.

```python
!git clone https://github.com/aadityaincode/RAG-Workplace-Assistant.git
%cd RAG-Workplace-Assistant
```

---

## Step 2: Install Dependencies

Install all Python packages listed in `requirements.txt`, then separately install `streamlit` and `pyngrok`. These two are not included in `requirements.txt` because they are only needed in the Colab/tunneling environment, not in a standard local setup.

```python
# Install core project dependencies
!pip install -r requirements.txt

# Install Streamlit (web app framework) and pyngrok (tunneling library)
!pip install streamlit pyngrok -q
```

---

## Step 3: Configure API Keys

This project requires two API credentials:

1. **Gemini API Key** — used to power the AI responses via Google's Gemini model
2. **ngrok Auth Token** — used to expose your locally running Streamlit app to a public URL

Both are stored as **Colab Secrets** so they are never hardcoded or exposed in your notebook.

### 3a. Get your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **"Get API key"** in the left sidebar
4. Click **"Create API key"** and copy the generated key

### 3b. Get your ngrok Auth Token

1. Go to [ngrok.com](https://dashboard.ngrok.com/signup) and create a free account
2. After signing in, navigate to **Your Authtoken** in the left sidebar
3. Copy your auth token

### 3c. Add both secrets to Colab

1. In your Colab notebook, click the **key icon** in the left sidebar to open **Secrets**
2. Click **"Add new secret"**
3. Add the following two secrets:

| Name | Value |
|---|---|
| `GOOGLE_API_KEY` | Your Gemini API key from Google AI Studio |
| `NGROK_AUTH_TOKEN` | Your auth token from ngrok dashboard |

4. Toggle **"Notebook access"** to ON for both secrets

### 3d. Authenticate ngrok

Once your secrets are saved, run the following cell to authenticate the ngrok client:

```python
from pyngrok import ngrok
from google.colab import userdata

# Retrieve the ngrok auth token from Colab Secrets
NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN')

if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("ngrok authenticated successfully.")
else:
    print("Error: NGROK_AUTH_TOKEN not found. Please follow Step 3c above.")
```

---

## Step 4: Ingest Documents and Launch the App

This step does three things in sequence:
1. Runs the ingestion script to populate the ChromaDB vector database with your workplace documents
2. Writes the Gemini API key to a `.env` file so the app can read it at runtime
3. Starts the Streamlit server and creates a public ngrok tunnel to access it

```python
from pyngrok import ngrok
from google.colab import userdata
import subprocess

# --- 1. Ingest documents into the vector database ---
# This reads all documents from the /data folder, generates embeddings,
# and stores them in ChromaDB. Only needs to be run once per session.
!python ingest.py

# --- 2. Write the Gemini API key to a .env file ---
# The app reads credentials from .env at startup using python-dotenv.
# We pull the key from Colab Secrets to avoid hardcoding sensitive values.
google_api_key = userdata.get('GOOGLE_API_KEY')

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in Colab Secrets. Please complete Step 3.")

with open('.env', 'w') as f:
    f.write(f'GOOGLE_API_KEY={google_api_key}\n')

print(".env file created successfully.")

# --- 3. Start the Streamlit app as a background process ---
# Popen launches the server without blocking the notebook cell.
# The app will run on localhost:8501 inside Colab's environment.
process = subprocess.Popen(
    ["streamlit", "run", "streamlit_app.py"],
    stdout=subprocess.DEVNULL,  # Suppress output to keep the notebook clean
    stderr=subprocess.DEVNULL
)

# --- 4. Open an ngrok tunnel to the Streamlit port ---
# Since Colab runs in an isolated VM, we use ngrok to forward
# traffic from a public HTTPS URL to the local port 8501.
public_url = ngrok.connect(8501)
print(f"\nYour app is live at: {public_url}")
print("Open the link above to access the Workplace Assistant.")
```

---

## Stopping the App

To shut down the Streamlit server and close the ngrok tunnel when you're done:

```python
# Terminate the Streamlit background process
process.terminate()

# Disconnect all active ngrok tunnels
ngrok.kill()

print("App stopped and tunnel closed.")
```

---

## Troubleshooting

**The ngrok URL shows an error page**
The Streamlit server may still be starting up. Wait 5–10 seconds after running Step 4 and then refresh the URL.

**`GOOGLE_API_KEY not found` error**
Make sure the secret name in Colab Secrets is exactly `GOOGLE_API_KEY` (case-sensitive) and that "Notebook access" is toggled on.

**`ERR_NGROK_108` or tunnel limit error**
Your free ngrok account already has an active tunnel open. Go to the [ngrok dashboard](https://dashboard.ngrok.com/) and terminate any existing tunnels, then re-run Step 4.

**ChromaDB or ingestion errors**
If the database appears corrupted, delete the existing ChromaDB directory and re-run the ingestion step:
```python
!rm -rf chroma_db
!python ingest.py
```