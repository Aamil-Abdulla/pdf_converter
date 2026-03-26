# 📄 RAG PDF Chatbot

An AI-powered PDF chatbot built with Retrieval Augmented Generation (RAG). Upload any PDF and ask questions about its content — powered by Google Gemini and LangChain.

## 🚀 Live Demo
[Deploy link here]

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python
- **AI/ML:** LangChain, Google Gemini API, HuggingFace Embeddings
- **Vector Store:** FAISS
- **Deployment:** Docker, Render
- **Frontend:** HTML, CSS, JavaScript

## ⚙️ How It Works
1. User uploads a PDF
2. Text is extracted and split into chunks
3. Chunks are converted to vectors and stored in FAISS
4. User asks a question
5. FAISS finds the most relevant chunks
6. Gemini answers the question using those chunks

## 🏃 How To Run Locally
1. Clone the repo
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   uvicorn main:app --reload
4. Open http://localhost:8000
5. Upload a PDF, enter your Gemini API key, ask questions

## 📁 Project Structure
├── main.py          # FastAPI endpoints
├── rag.py           # RAG pipeline functions
├── index.html       # Frontend UI
├── Dockerfile       # Container config
└── requirements.txt # Dependencies

## 🔑 Get Gemini API Key
Get a free key at https://aistudio.google.com