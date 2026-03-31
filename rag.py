from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(file_path):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file_path, strict=False)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    print(f"Extracted text length: {len(text)}")
    print(f"First 200 chars: {text[:200]}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    model1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

    prompt = PromptTemplate(
        template="""
Context: {context}
Question: {question}

If the relevant topic cannot be found in the context, strictly say "I'm sorry, but I don't have information on that topic."
Answer:
""",
        input_variables=["context", "question"]
    )

    chain = prompt | model1 | StrOutputParser()
    result = chain.invoke({"context": context, "question": user_question})
    return result