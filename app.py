import streamlit as st
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import os

# Ensure the event loop is handled properly
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Streamlit App Title
st.title("LangChain FAISS Search Demo")

# Load FAISS index
INDEX_PATH = "faiss_index/index.faiss"

if os.path.exists(INDEX_PATH):
    st.success("FAISS index found. Loading...")
    index = faiss.read_index(INDEX_PATH)
else:
    st.warning("FAISS index not found. Creating a new one...")
    index = faiss.IndexFlatL2(768)  # Example for 768-d embeddings

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings()

# Input for Query
query = st.text_input("Enter your query:")

if query:
    st.write("Searching...")
    query_embedding = embeddings.embed_query(query)
    
    # Perform FAISS search
    D, I = index.search([query_embedding], k=5)
    
    st.write("Search Results:")
    for i, idx in enumerate(I[0]):
        st.write(f"{i+1}. Result {idx} (Distance: {D[0][i]:.4f})")

