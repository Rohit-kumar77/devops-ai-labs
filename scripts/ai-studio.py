import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
st.header("🌐 Hosted PDF Chatbot (Google Gemini)")

# ✅ Google Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    google_api_key="xxxxxxxxxxxx"
)

# ✅ Local embeddings (still best choice)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

with st.sidebar:
    st.title("📄 Upload PDF")
    file = st.file_uploader("Choose PDF", type="pdf")

vector_store = None

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

    st.success(f"✅ PDF Indexed ({len(chunks)} chunks)")

user_question = st.text_input("💬 Ask a question")

if user_question:

    if vector_store:
        docs = vector_store.similarity_search(user_question, k=4)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = PromptTemplate(
            template="""
Answer ONLY from the context below.
If answer not present, say "I don't know".

Context:
{context}

Question:
{question}
""",
            input_variables=["context", "question"]
        )

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "question": user_question
        })

    else:
        response = llm.invoke(user_question)

    st.subheader("📌 Answer")
    st.write(response)
