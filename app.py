import streamlit as st
import torch
import pypdf
import os
import pickle
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Set Page Title
st.set_page_config(page_title="PDF Text Extractor & Search", layout="wide")
st.title("📄 PDF Text Extractor & Search with Llama 3.2")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    def extract_text_from_pdf(pdf_file):
        reader = pypdf.PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text

    pdf_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text:")
    st.text_area("PDF Content", pdf_text, height=300)

    # Load embeddings model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_text(pdf_text)
    
    # Convert docs to embeddings & store in FAISS
    vector_store = FAISS.from_texts(docs, embedding_model)
    vector_store.save_local("faiss_index")
    
    # Load FAISS index
    vector_store = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Load Llama 3.2 Model
    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    # Define LLM pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Define Retrieval Augmented Generation (RAG) pipeline
    rag_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=llm_pipeline),
        retriever=retriever
    )
    
    # User Query Input
    query = st.text_input("Enter your search query:")
    if query:
        response = rag_chain.invoke(query)
        formatted_output = response['result'].replace("•", "-").replace("\n\n", "\n").strip()
        st.subheader("Response:")
        st.write(formatted_output)

    # Summarization
    if st.button("Summarize PDF Content"):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(pdf_text, max_length=150, min_length=50, do_sample=False)
        st.write("🔍 **Summary:**")
        st.write(summary[0]['summary_text'])
