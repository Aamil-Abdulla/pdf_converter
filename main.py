from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from rag import extract_text, get_text_chunks, get_vector_store, get_answer
from pydantic import BaseModel
import os
app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile , api_key: str = Form(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    contents = await file.read()
    
 
    with open("uploaded.pdf", "wb") as f:
        f.write(contents)
    

    text = extract_text("uploaded.pdf")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Is it a scanned image?")
    
    chunks = get_text_chunks(text)
    get_vector_store(chunks, api_key)
    return {"message": "PDF uploaded and processed successfully."}

@app.get("/")
async def home():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

class AskRequest(BaseModel):
    question: str
    api_key: str

@app.post("/ask")
async def ask_question(request: AskRequest):
    if not os.path.exists("faiss_index"):
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    
    answer = get_answer(request.question, request.api_key)
    return {"answer": answer}