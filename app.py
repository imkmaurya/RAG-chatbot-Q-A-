import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from loader import load_doc
from text_splitter import split_documents
from embedding import make_embedding
from vector_store import make_vectorstore
from retriver import retreival
from llm import get_chain

load_dotenv()

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")

st.title("📄 Document Q&A Chatbot")
st.write("Upload a file and ask questions about it.")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False

# --- Sidebar: upload & process ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "csv", "txt"])

    if st.button("Process Document"):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Processing document..."):
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                docs = load_doc(tmp_path)
                splitted_text = split_documents(docs)
                embedding = make_embedding()
                db = make_vectorstore(splitted_text, embedding)
                retriever = retreival(db)
                chain = get_chain()

                st.session_state.retriever = retriever
                st.session_state.chain = chain
                st.session_state.doc_ready = True
                st.session_state.messages = []

                os.remove(tmp_path)

            st.success(f"'{uploaded_file.name}' processed! You can start chatting now.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat input ---
if not st.session_state.doc_ready:
    st.info("Upload and process a document to start chatting.")

user_input = st.chat_input("Ask a question...", disabled=not st.session_state.doc_ready)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = st.session_state.retriever.invoke(user_input)
            context = "\n\n".join(doc.page_content for doc in docs)
            response = st.session_state.chain.invoke({
                "context": context,
                "question": user_input
            })
            answer = response if isinstance(response, str) else str(response)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})