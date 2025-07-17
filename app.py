import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Changed import from ChatGroq to ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env variables before using them
load_dotenv()
# Changed environment variable from GROQ_API_KEY to GOOGLE_API_KEY
google_api_key = os.getenv("GOOGLE_API_KEY")

# Validate key
if not google_api_key:
    st.error("GOOGLE_API_KEY is missing. Please check your `.env` file.")
    st.stop()

# Initialize Google's LLM after loading the key
# Replaced ChatGroq with ChatGoogleGenerativeAI and specified a Gemini model
llm = ChatGoogleGenerativeAI(
    google_api_key=google_api_key,
    model="gemini-1.5-flash-latest", # Using one of Google's most capable models
    temperature=0.7
)

# Streamlit layout
st.set_page_config(page_title="PDF Summarizer", layout="centered")
# Updated the title to reflect the change in backend
st.title("📄 PDF Summarizer using LangChain and Google AI")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Step 1: Extract text from PDF
    try:
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        if not raw_text.strip():
            st.error("No extractable text found in the PDF.")
            st.stop()

        st.success("Text extracted from PDF")

        # Step 2: Chunk the text (This part remains the same)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(raw_text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        st.write(f"🔹 Total chunks: {len(documents)}")

        # Step 3: Prompt and summarize (The chain structure remains the same)
        prompt = PromptTemplate.from_template(
            "Summarize the following content concisely:\n\n{text}\n\nSummary:"
        )
        chain = prompt | llm | StrOutputParser()

        with st.spinner("Summarizing with Google AI..."):
            # Note: For very large documents, you might need a more advanced
            # summarization strategy like MapReduce or Refine.
            # For this example, we'll summarize the beginning of the text.
            summary = chain.invoke({"text": raw_text})

        st.subheader("Summary")
        st.success(summary)

    except Exception as e:
        st.error(f"An error occurred: {e}")

