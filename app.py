import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pdf2image import convert_from_path
import pytesseract
import os
import tempfile
import fitz  # PyMuPDF

# --- Streamlit UI ---
st.set_page_config(page_title="📄 Contract Q&A Chatbot")
st.title("📄 Contract Q&A Chatbot with GPT-4")

uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

import os
os.environ["STREAMLIT_DISABLE_TELEMETRY"] = "1"

# Patch: Stop Streamlit from writing machine_id_v4 entirely
import streamlit.runtime.metrics_util
streamlit.runtime.metrics_util._get_machine_id_v4 = lambda: "patched-machine-id"

openai_api_key = os.environ.get("OPENAI_API_KEY")


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def process_pdf_to_chunks(pdf_path):
    images = convert_from_path(pdf_path)
    docs = []
    for i, image in enumerate(images[:6]):  # Limit for quick testing
        page_text = pytesseract.image_to_string(image)
        chunks = chunk_text(page_text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"page": i + 1}))
    return docs


def setup_qa_chain(docs):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embedding=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        docs = process_pdf_to_chunks(tmp_path)
        qa_chain = setup_qa_chain(docs)
        st.success("PDF processed!")

    question = st.text_input("Ask a question:")
    if question:
        response = qa_chain.invoke({"query": question})
        st.subheader("Answer")
        st.write(response["result"])

        st.subheader("Sources")
        for doc in response["source_documents"]:
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Page {page}:**")
            st.code(doc.page_content[:400])
