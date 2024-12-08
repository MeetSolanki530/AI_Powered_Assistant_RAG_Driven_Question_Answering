import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Pinecone initialization
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please set it in your .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY, ssl_verify=False)
index = pc.Index("chatbot")

# Streamlit app title
st.title("AI-Powered Chatbot")

# Initialize Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile")

# Embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are an interactive friendly chatbot.
    Don't give answer if it's not in provided documents.
    Answer the question based on the provided context only. Provide a proper and detailed response. Avoid redundancy of sentence in Answer. Do not use the words in starting of answer such as "According to the provided context".
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to process and upload PDFs
def process_and_upload_pdfs(uploaded_files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []

    for uploaded_file in uploaded_files:
        # Save PDF locally
        pdf_path = os.path.join("uploaded_pdfs", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and split PDF content
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(text_splitter.split_documents(docs))

    # Upload documents to Pinecone
    PineconeVectorStore.from_documents(
        documents,
        embedding=embedding_model,
        index_name="chatbot",
    )
    st.success("PDFs successfully uploaded to Pinecone!")

# Function to initialize Pinecone retriever
def initialize_retriever():
    return PineconeVectorStore(index_name="chatbot", embedding=embedding_model)

# Sidebar for PDF upload
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", accept_multiple_files=True, type=["pdf"]
)

if st.sidebar.button("Upload PDFs"):
    if uploaded_files:
        if not os.path.exists("uploaded_pdfs"):
            os.makedirs("uploaded_pdfs")
        with st.spinner("Uploading and processing PDFs..."):
            process_and_upload_pdfs(uploaded_files)
    else:
        st.sidebar.warning("Please upload at least one PDF file.")

# User input for question
input_prompt = st.text_input("Enter your question based on the documents:")

if st.button("Search"):
    if not input_prompt.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching response..."):
            # Initialize retriever and create retrieval chain
            retriever = initialize_retriever().as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Fetch the response
            response = retrieval_chain.invoke({"input": input_prompt})
            st.write("*Response:*", response["answer"])

            # Show relevant document chunks
            with st.expander("Relevant Document Chunks"):
                for doc in response["context"]:
                    st.write(doc.page_content)
                    st.write("---")