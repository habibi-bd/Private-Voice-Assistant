import os
import ollama
import tempfile
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def process_file(file_source, is_path=False):
    """Loads and indexes the document into a vector store."""
    if is_path:
        loader = PyPDFLoader(file_source) if file_source.endswith(".pdf") else TextLoader(file_source)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_source.name)[1]) as tmp:
            tmp.write(file_source.getvalue())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path) if tmp_path.endswith(".pdf") else TextLoader(tmp_path)
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    if not is_path: os.remove(tmp_path)
    return vectorstore

def ask_rag_llm(query, vectorstore):
    """Queries the local vector store and returns a clean LLM response."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    response = ollama.chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": "You are a private assistant. Answer ONLY using the context. Keep it short. No markdown."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    )
    return response['message']['content'].replace("*", "").strip()