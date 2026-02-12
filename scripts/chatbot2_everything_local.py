import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local AI Assistant", layout="wide")
st.header("🧠 Local AI Assistant (Optimized for 8GB RAM)")

# ✅ PERFECT MODEL FOR YOUR MACHINE
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

with st.sidebar:
    st.title("📄 Optional PDF Mode")
    file = st.file_uploader("Upload PDF", type="pdf")

vector_store = None

if file:
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # slightly reduced for low RAM
        chunk_overlap=120
    )

    chunks = splitter.split_text(text)

    vector_store = FAISS.from_texts(chunks, embeddings)

    st.success(f"✅ PDF Indexed ({len(chunks)} chunks)")

user_question = st.text_input("💬 Ask anything (PDF / DevOps / Tech / General)")

if user_question:

    if vector_store:
        docs = vector_store.similarity_search(user_question, k=3)

        context = "\n\n".join(d.page_content for d in docs)

        rag_prompt = PromptTemplate(
            template="""
Answer ONLY from the context below.
If answer not present, say NOT_FOUND.

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )

        chain = rag_prompt | llm | StrOutputParser()

        rag_response = chain.invoke({
            "context": context,
            "question": user_question
        })

        if "NOT_FOUND" not in rag_response:
            st.subheader("📄 Document Answer")
            st.write(rag_response)
        else:
            st.subheader("🌍 General Answer")
            st.write(llm.invoke(user_question))

    else:
        st.subheader("🌍 Answer")
        st.write(llm.invoke(user_question))
