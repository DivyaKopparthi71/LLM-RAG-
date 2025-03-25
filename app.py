import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hugging Face Authentication
HUGGINGFACE_TOKEN = "your_correct_hf_token"
os.environ["HF_HOME"] = "/home/adminuser/.cache/huggingface"
os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

# Use Correct Model
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Change if needed

# Load Model & Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("LLM Chatbot with Llama-3")
query = st.text_input("Enter your question:")
if query:
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
