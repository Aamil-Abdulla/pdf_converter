from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text(file_path):
    text = ""
    Pdf_reader = PyPDF2.PdfReader(file_path)
    for page in Pdf_reader.pages:
        text += page.extract_text()
    print(f"Extracted text length: {len(text)}")  # add this
    print(f"First 200 chars: {text[:200]}")        # add this
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(user_question, api_key):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    
    # Try adding the 'models/' prefix explicitly if you haven't yet
    model1 = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash", 
    google_api_key=api_key,
    convert_system_message_to_human=True # Helpful for older library versions
    )
    
    prompt_template = """
    Context: {context}
    Question: {question}
    
    If the relevant topic cannot be found in the context, strictly say "I'm sorry, but I don't have information on that topic."
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = create_stuff_documents_chain(llm=model1, prompt=prompt)
    
    # This calls the Google API
    result = chain.invoke({"context": docs, "question": user_question})
    return result