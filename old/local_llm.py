import re
import sounddevice as sd
import scipy.io.wavfile as wav
import pyttsx3
from faster_whisper import WhisperModel
import ollama
import numpy as np


# Load Whisper model
whisper = WhisperModel("tiny.en")

# Ollama model
model_name = "gemma3:latest"



def ask_llm(prompt):
    if not prompt:
        return ""
    try:
        # The System Message forces the AI to stop sending links/markdown
        system_instruction = (
            "You are a concise voice assistant. Give short, conversational answers. "
            "Do not use markdown, bolding (**), bullet points, or provide URLs/links. "
            "Provide only the spoken text."
        )
        
        response = ollama.chat(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
        )
        
        raw_text = response['message']['content']
        
        # --- CLEANING PROCESS ---
        # 1. Remove Markdown bold/italic symbols (e.g., **text** -> text)
        clean_text = raw_text.replace("*", "")
        
        # 2. Remove URLs (http/https links)
        clean_text = re.sub(r'http\S+', '', clean_text)
        
        # 3. Remove Disclaimer blocks (optional but helpful for stock queries)
        if "Disclaimer" in clean_text:
            clean_text = clean_text.split("Disclaimer")[0]

        return clean_text.strip()
        
    except Exception as e:
        return f"Error: {e}"