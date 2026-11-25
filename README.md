# AskMyDocs — RAG Knowledge Assistant

A local RAG (Retrieval-Augmented Generation) document assistant that lets you upload PDFs and ask questions, with source-grounded answers and page-level citations — no paid LLM APIs required.

## Features

- PDF ingestion and text extraction (PyMuPDF)
- Document chunking with 384-dimensional embeddings
- Semantic search using FAISS (top-K vector similarity)
- Page-level citation metadata in answers
- Runs fully locally via Ollama
- Retrieval evaluation: Hit@K, citation accuracy, groundedness, latency

## Tech Stack

- **Backend:** Python, FastAPI
- **Retrieval:** FAISS, Sentence Transformers
- **PDF:** PyMuPDF
- **LLM:** Ollama (local models)
- **Evaluation:** Hit@K, citation accuracy, groundedness, latency

## How It Works

1. Upload a PDF → text extracted with PyMuPDF
2. Text chunked and embedded into 384-dim vectors
3. Embeddings stored in FAISS index
4. Query embedded → top-K similar chunks retrieved
5. Retrieved chunks + query sent to local LLM (Ollama)
6. Answer returned with page-level citations
7. Evaluation separates retrieval failures from generation failures

## Setup

```bash
git clone https://github.com/DineshKarri/AskMyDocs.git
cd AskMyDocs
pip install -r requirements.txt
# Install Ollama and pull a model
ollama pull llama3
uvicorn main:app --reload
