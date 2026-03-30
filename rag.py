import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

def extract_text(file_path):
    text = ""
    # Using 'rb' mode is safer for PDF reading
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f, strict=False)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:  
                text += extracted
    return text

def get_text_chunks(text):
    # Reduced chunk size slightly for better retrieval precision
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks, api_key):
    # FIX: Using Google Embeddings to save Render RAM
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(user_question, api_key):
    # FIX: Must use the same embedding model for loading that you used for saving
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Load the index
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=3) # Get top 3 chunks

    # Use Gemini Flash for speed
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

    prompt_template = """
    Context: {context}
    Question: {question}

    If the relevant topic cannot be found in the context, strictly say "I'm sorry, but I don't have information on that topic."
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # Invoke the chain
    result = chain.invoke({"context": docs, "question": user_question})
    return result