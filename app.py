import streamlit as st
import os
import time
from streamlit_mic_recorder import speech_to_text
from audio_utils import speak_text
from rag_engine import process_file, ask_rag_llm

# --- CONFIGURATION ---
DEFAULT_FILE = "faq_llm.pdf"
st.set_page_config(page_title="Voice-Ready AI Assistant", page_icon="🎙️", layout="wide")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("⚙️ Controls")
    use_voice_out = st.toggle("Voice Output (TTS)", value=False)
    
    st.divider()
    st.subheader("Voice Input (STT)")
    voice_input = speech_to_text(start_prompt="Click to Speak 🎙️", stop_prompt="Stop Recording ⏹️", language='en', key='STT')
    
    st.divider()
    st.subheader("Document")
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build/Load", use_container_width=True):
            with st.spinner("Processing..."):
                if uploaded_file: 
                    st.session_state.vectorstore = process_file(uploaded_file)
                elif os.path.exists(DEFAULT_FILE): 
                    st.session_state.vectorstore = process_file(DEFAULT_FILE, is_path=True)
                else: 
                    st.error("No file found.")
                st.success("Ready!")
                
    with c2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- MAIN CHAT UI ---
st.title("🤖 Voice-Enabled Private Assistant")

if st.session_state.vectorstore:
    active_prompt = None
    if voice_input:
        active_prompt = voice_input
    if manual_prompt := st.chat_input("Type your question..."):
        active_prompt = manual_prompt

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if active_prompt:
        st.session_state.chat_history.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                reply = ask_rag_llm(active_prompt, st.session_state.vectorstore)
                st.markdown(reply)
                if use_voice_out:
                    speak_text(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
else:
    st.info(f"💡 Upload a file or click 'Build/Load' in the sidebar to use **{DEFAULT_FILE}**.")