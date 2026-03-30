# faq_assistant.py
import ollama
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Optional: for TTS / microphone
# from voice_loop import record_audio, speech_to_text, speak
# If you don't have mic setup, you can simulate input with input()

FILE_PATH = "faq_llm.pdf"

def prepare_knowledge_base(file_path):
    """Load document, split it, and create a vector store."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please place the file in this folder.")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./chroma_db"
    )
    return vectorstore

def ask_rag_llm(user_query, vectorstore):
    """Query the vector store and call the Ollama LLM."""
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])

    system_prompt = (
        "You are a private assistant. Answer the user's question ONLY using the provided context. "
        "If the answer is not in the context, say 'I do not have that information in my files.' "
        "Keep it short for voice output. No markdown."
    )

    full_prompt = f"Context: {context}\n\nQuestion: {user_query}"

    response = ollama.chat(
        model="gemma3:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )

    reply = response['message']['content']
    return reply.replace("*", "").strip()

def main():
    # --- Build Knowledge Base ---
    print("Step 1: Building knowledge base (may take a moment)...")
    vectorstore = prepare_knowledge_base(FILE_PATH)
    print("Knowledge base ready!")

    # --- Voice Assistant Loop ---
    print("Step 2: Start asking questions. Type 'exit' to quit.")
    while True:
        try:
            # Replace input() with real speech capture if you want mic
            user_text = input("You: ")
            
            if user_text.lower() in ["exit", "quit"]:
                print("Shutting down...")
                break

            if user_text.strip() == "":
                continue  # ignore empty input

            # Query the RAG LLM
            reply = ask_rag_llm(user_text, vectorstore)
            print(f"Assistant: {reply}")

            # Optional: speak(reply) if you have TTS
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()