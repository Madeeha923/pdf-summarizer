import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load .env variables before using them
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate key
if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Please check your `.env` file.")
    st.stop()

# Initialize LLM after loading the key
llm = ChatGroq(
    api_key=groq_api_key,  
    model_name="gemma2-9b-it",  
    temperature=0.7
)

# Streamlit layout
st.set_page_config(page_title="PDF Summarizer", layout="centered")
st.title("📄 PDF Summarizer using LangChain and Groq")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Step 1: Extract text from PDF
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text

    if not raw_text.strip():
        st.error("No extractable text found in the PDF.")
        st.stop()

    st.success("✅ Text extracted from PDF")

    # Step 2: Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    st.write(f"🔹 Total chunks: {len(documents)}")

    # Step 3: Prompt and summarize
    prompt = PromptTemplate.from_template(
        "Summarize the following content:\n\n{text}\n\nSummary:"
    )
    chain = prompt | llm | StrOutputParser()

    with st.spinner("Summarizing..."):
        summary = chain.invoke({"text": raw_text[:3000]})  # summarizing sample chunk

    st.subheader(" Summary")
    st.success(summary)
