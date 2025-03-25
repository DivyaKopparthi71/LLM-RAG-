import streamlit as st
import torch
import pypdf
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Page Title
st.set_page_config(page_title="PDF Text Extractor & Search", layout="wide")
st.title("📄 PDF Text Extractor & Search with Llama 3.2")

# Load Model
@st.cache_resource()
def load_model():
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    def extract_text_from_pdf(pdf_file):
        reader = pypdf.PdfReader(pdf_file)
        return "".join(page.extract_text() or "" for page in reader.pages)

    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text:")
    st.text_area("PDF Content", pdf_text, height=300)

    # Load embeddings model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_text(pdf_text)
    
    # Store in FAISS
    vector_store = FAISS.from_texts(docs, embedding_model)

    # Save FAISS Index
    index_path = "faiss_index"
    vector_store.save_local(index_path)
    st.success("PDF content indexed successfully!")
