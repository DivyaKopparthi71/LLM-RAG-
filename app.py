import streamlit as st
import torch
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import pypdf
import os

# Set Page Title
st.set_page_config(page_title="PDF Text Extractor & Search", layout="wide")

# Title
st.title("📄 PDF Text Extractor & Search")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    def extract_text_from_pdf(pdf_file):
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    # Process PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text:")
    st.text_area("PDF Content", pdf_text, height=300)

    # Generate Embeddings using Sentence Transformers
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    st.subheader("Generating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Store in FAISS Vector Database
    db = FAISS.from_texts([pdf_text], embeddings)

    # User Query Input
    query = st.text_input("Enter your search query:")

    if query:
        # Perform Similarity Search
        results = db.similarity_search(query, k=3)
        st.subheader("Search Results:")
        for i, result in enumerate(results):
            st.write(f"🔹 **Match {i+1}:** {result.page_content}")

# Transformer-based Text Summarization
st.subheader("Summarization using Transformers")
if st.button("Summarize PDF Content"):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(pdf_text, max_length=150, min_length=50, do_sample=False)
    st.write("🔍 **Summary:**")
    st.write(summary[0]['summary_text'])

# Run using: streamlit run app.py
