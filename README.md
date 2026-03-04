# VERITUS AI Service

AI-powered answer evaluation microservice built with FastAPI, FAISS, and a local LLM (Phi-3 via Ollama).

## Features
- Retrieval-Augmented Generation (RAG)
- Semantic search using FAISS
- Local LLM evaluation
- Score + feedback generation
- REST API endpoint for backend integration

## Architecture
User → Frontend → Backend → AI Service → FAISS + LLM → Score + Feedback

## API Endpoint

POST /evaluate

Example request:

{
  "question": "What is Docker?",
  "answer": "Docker is used for containers.",
  "testType": "TECHNICAL",
  "difficulty": "MEDIUM"
}

Example response:

{
  "score": 7,
  "feedback": "Correct answer but lacks detail."
}

## Technologies Used

- FastAPI
- FAISS
- Sentence Transformers
- Ollama (Phi-3)
- Python

## How to Run

Install dependencies:

pip install -r requirements.txt

Run server:

python run.py

Open API docs:

http://localhost:8000/docs
