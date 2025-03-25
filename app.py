import os
import pickle
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Set Streamlit Page Config
st.set_page_config(page_title="LLaMA-3.2 Chatbot", page_icon="🤖", layout="wide")

# Hugging Face Authentication (if needed)
HUGGINGFACE_TOKEN = "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"
os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

# Title and Styling
st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">🦙 LLaMA-3.2 Chatbot with RAG</h1>
    <p style="text-align:center;">Ask any question and get intelligent responses from LLaMA 3.2!</p>
    """,
    unsafe_allow_html=True,
)

# Load Model & Tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Load FAISS Index & Embeddings
@st.cache_resource()
def load_faiss():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_faiss()
retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Limit retrieved docs

# Define LLM Pipeline
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

# Define RAG Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=llm_pipeline),
    retriever=retriever
)

# Query Input
query = st.text_input("🔍 Ask your question:", placeholder="E.g., What are Consumer Benefits?")

if st.button("Generate Response"):
    if query:
        with st.spinner("🤖 Generating response..."):
            response = rag_chain.invoke(query)
            formatted_output = response['result'].replace("•", "-").replace("\n\n", "\n").strip()
            
            # Display response
            st.markdown(
                f"""
                <div style="background:#f9f9f9; padding:15px; border-radius:10px;">
                <h4 style="color:#333;">💡 Answer:</h4>
                <p>{formatted_output}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Save output as a .pkl file
            with open("formatted_output.pkl", "wb") as f:
                pickle.dump(formatted_output, f)

            st.success("✅ Response saved successfully in `formatted_output.pkl`")
    else:
        st.warning("⚠️ Please enter a question before generating a response.")
