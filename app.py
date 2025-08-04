# app.py

import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import pytesseract
from pdf2image import convert_from_path
import tempfile
import os

# --- Streamlit UI ---
st.set_page_config(page_title="📄 Contract Q&A Chatbot")
st.title("📄 Contract Q&A Chatbot with GPT-4")

uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

# --- OCR + Chunking ---
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def process_pdf_to_chunks(pdf_path):
    convert_from_path(pdf_path, poppler_path="/usr/bin")
    st.info(f"Found {len(images)} pages.")

    docs = []
    progress = st.progress(0)

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        chunks = chunk_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"page": i + 1}))
        progress.progress((i + 1) / len(images))
    
    return docs

# --- Embedding & QA Setup ---
def setup_qa_chain(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- Main App Logic ---
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Reading and processing the PDF..."):
        docs = process_pdf_to_chunks(tmp_path)
        qa_chain = setup_qa_chain(docs)

    st.success("✅ PDF processed successfully!")

    question = st.text_input("Ask a question about the contract:")
    if question:
        with st.spinner("Searching for the answer..."):
            response = qa_chain.invoke({"query": question})
            st.subheader("🧠 Answer")
            st.write(response["result"])

            st.subheader("📄 Source(s):")
            for doc in response["source_documents"]:
                page = doc.metadata.get("page", "?")
                st.markdown(f"**📄 Page {page}:**")
                st.code(doc.page_content[:500].strip())
