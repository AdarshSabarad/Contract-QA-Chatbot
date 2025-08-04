# 📄 Contract Q&A Chatbot

An AI-powered chatbot that lets users upload any contract or scanned PDF and ask natural language questions like:

> _"What is the termination clause?"_  
> _"Is there a price escalation policy?"_

Built using:
- 🔍 OCR (Tesseract) for scanned documents
- ✂️ LangChain for chunking and retrieval
- 🤖 GPT-4 via OpenAI for question answering
- 🧠 FAISS for vector-based memory
- 🖥️ Streamlit for a simple, user-friendly interface

---

## 🚀 Features

- Upload any **contract PDF**, including **scanned documents**
- Ask free-form questions about clauses, terms, or policies
- Get answers **powered by GPT-4**
- View the exact source text with **page number citation**
- Streamlit-based interface — easy to use, share, and deploy

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/contract-qa-chatbot.git
cd contract-qa-chatbot
pip install -r requirements.txt
