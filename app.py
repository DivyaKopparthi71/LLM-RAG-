import os
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio

# Fix RuntimeError: no running event loop
asyncio.set_event_loop(asyncio.new_event_loop())

# Hugging Face Authentication (if private model)
HUGGINGFACE_TOKEN = "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"
os.environ["HF_HOME"] = "/home/adminuser/.cache/huggingface"
os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

# Load Model & Tokenizer
MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"  # Check if this exists
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
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

