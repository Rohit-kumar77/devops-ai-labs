import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PDF Chatbot (Offline)", layout="wide")
st.header("📄 PDF Chatbot (Offline + No Quota + Low RAM)")

# -----------------------------
# Local Embeddings (FREE)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------
# Local LLM (Low RAM Safe)
# -----------------------------
llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📂 Upload PDF")
    file = st.file_uploader("Choose a PDF file", type="pdf")

# -----------------------------
# Main Logic
# -----------------------------
if file:
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embeddings)

    st.success(f"✅ PDF processed successfully ({len(chunks)} chunks)")

    user_question = st.text_input("💬 Ask a question about the document")

    if user_question:
        docs = vector_store.similarity_search(user_question, k=4)

        prompt = PromptTemplate(
            template="""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": "\n\n".join(d.page_content for d in docs),
            "question": user_question
        })

        st.subheader("📌 Answer")
        st.write(response)
