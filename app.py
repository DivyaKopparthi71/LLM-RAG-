import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hugging Face Authentication
HUGGINGFACE_TOKEN = "your_hf_token"  # Replace with your actual token
os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

# Load a Smaller LLaMA Model (8B version)
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # Use float16 for GPU, float32 for CPU

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map="auto"
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("LLaMA-3 Chatbot")
query = st.text_input("Enter your question:")
if query:
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
