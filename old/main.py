import os
from prepare_knowledge import prepare_knowledge_base
from ask_rag_llm import ask_rag_llm
from voice_loop import record_audio, speech_to_text, speak

def main():
    # --- CONFIGURATION ---
    # Path to your private PDF/TXT file
    FILE_PATH = "faq_llm.pdf" 
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found. Please place the file in this folder.")
        return

    # --- INITIALIZATION ---
    print("Step 1: Building knowledge base (this may take a moment)...")
    # This creates the Chroma vector store from your file
    vectorstore = prepare_knowledge_base(FILE_PATH)
    print(vectorstore)
    print("Knowledge base ready!")

    print("Step 2: Starting Voice Assistant. Speak now!")
    
    # --- MAIN LOOP ---
    while True:
        try:
            # 1. Capture Audio
            record_audio(duration=5)
            
            # 2. Convert to Text
            user_text = speech_to_text()
            
            if user_text:
                print(f"You asked: {user_text}")
                
                # 3. Get RAG Response (using your ask_rag_llm.py file)
                # It will only answer from the document!
                reply = ask_rag_llm(user_text, vectorstore)
                
                print(f"Assistant: {reply}")
                
                # 4. Speak the answer
                speak(reply)
            else:
                print("...") # Silent loop when no speech detected
                
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
    