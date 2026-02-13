# test_env.py
import os
from pathlib import Path
from dotenv import load_dotenv

print(f"\nCurrent directory: {os.getcwd()}")
print(f".env exists: {os.path.exists('.env')}")

# Load and check
load_dotenv(override=True)
key = os.getenv("GOOGLE_API_KEY")

if key:
    print(f"\nLoaded key: {key[:10]}...{key[-4:]} (length: {len(key)})")
else:
    print("\nERROR: Key not loaded")