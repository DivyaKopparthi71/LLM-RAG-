import streamlit as st
import pickle
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Set page config
st.set_page_config(page_title="PDF AI Assistant", layout="wide")
st.title("📘 AI-Powered PDF Assistant")
st.markdown("Upload a PDF and enter your query below.")

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    model_name = "meta-llama/Llama-3B-Instruct"  # Ensure correct model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()
if model is None or tokenizer is None:
    st.stop()

st.success("Llama 3 Model Loaded Successfully!")

# File Uploader
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and Process PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Convert documents into embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "faiss_index"

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embedding_model)
    else:
        vector_store = FAISS.from_documents(docs, embedding_model)
        vector_store.save_local(index_path)

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Define LLM pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # Define RAG pipeline
    rag_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=llm_pipeline),
        retriever=retriever
    )

    # Streamlit Input Box
    query = st.text_input("🔍 Enter your query:", "What are Consumer Benefits?")

    if st.button("Get Response"):
        with st.spinner("Processing..."):
            try:
                response = rag_chain.invoke(query)
                formatted_output = response['result'].replace("•", "-").replace("\n\n", "\n").strip()

                # Save response
                with open("formatted_output.pkl", "wb") as f:
                    pickle.dump(formatted_output, f)

                st.subheader("📜 Response:")
                st.success(formatted_output)
            except Exception as e:
                st.error(f"Error generating response: {e}")
