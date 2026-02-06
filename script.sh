#!/bin/bash

echo "Setting up Python environment for RAG Workplace Assistant..."

# Remove old venv if it exists
if [ -d "venv" ]; then
  echo "Existing venv found. Removing it..."
  rm -rf venv
fi

# Create fresh virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo ""
echo "Environment ready."
echo "Activate it with:"
echo "source venv/bin/activate"
echo ""
echo "Then run:"
echo "python app.py"