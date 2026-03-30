from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def prepare_knowledge_base(file_path):
    # 1. Load the document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    docs = loader.load()

    # 2. Split into chunks (so the LLM doesn't get overwhelmed)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Create Vector Store (This stores the data locally in a folder 'db')
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OllamaEmbeddings(model="nomic-embed-text"), # Great for local RAG
        persist_directory="./chroma_db"
    )
    return vectorstore