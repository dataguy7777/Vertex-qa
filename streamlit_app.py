# app.py

import streamlit as st
import os
import uuid
import io
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
from typing import List, Dict, Tuple

# =========================
# Configuration and Setup
# =========================

# Load environment variables from .env file
load_dotenv()

# =========================
# Streamlit Page Configuration
# =========================

# Set Streamlit page configuration as the very first Streamlit command
st.set_page_config(
    page_title="📄 RAG App with Qdrant Cloud",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Retrieve Environment Variables
# =========================

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g., "https://your-instance.qdrant.io"

# =========================
# Validate Environment Variables
# =========================

if not all([QDRANT_API_KEY, QDRANT_URL]):
    st.error("Please ensure that QDRANT_API_KEY and QDRANT_URL are set in the .env file.")
    st.stop()

# =========================
# Initialize Qdrant Client
# =========================

try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
except Exception as e:
    st.error(f"Failed to connect to Qdrant Cloud: {e}")
    st.stop()

# Define Qdrant collection name
COLLECTION_NAME = "documents"

# Define embedding model parameters
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # A lightweight, efficient model
VECTOR_SIZE = 384  # Dimension size for all-MiniLM-L6-v2

# =========================
# Initialize Sentence Transformer Model
# =========================

@st.cache_resource
def load_embedding_model():
    """
    Loads the pre-trained SentenceTransformer model for generating embeddings.

    Returns:
        SentenceTransformer: The loaded embedding model.

    Example:
        model = load_embedding_model()
    """
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

embedding_model = load_embedding_model()

# =========================
# Utility Functions
# =========================

def initialize_qdrant_collection():
    """
    Initializes the Qdrant collection with the specified schema.
    If the collection already exists, it will not be recreated.

    Example:
        initialize_qdrant_collection()

    Output:
        None
    """
    try:
        existing_collections = qdrant_client.get_collections().collections
        if COLLECTION_NAME not in [col.name for col in existing_collections]:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            st.sidebar.success(f"Qdrant collection '{COLLECTION_NAME}' created successfully.")
        else:
            st.sidebar.info(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        st.sidebar.error(f"Error initializing Qdrant collection: {e}")
        st.stop()

def extract_text_from_pdf(file: io.BytesIO) -> List[Tuple[int, str]]:
    """
    Extracts text from a PDF file on a per-page basis.

    Args:
        file (io.BytesIO): The uploaded PDF file.

    Returns:
        List[Tuple[int, str]]: A list of tuples containing page numbers and extracted text.

    Example:
        pages = extract_text_from_pdf(uploaded_pdf)
    """
    try:
        pages = []
        laparams = LAParams()
        with tempfile.TemporaryFile() as temp_file:
            extract_text_to_fp(file, temp_file, laparams=laparams, output_type='text', codec=None)
            temp_file.seek(0)
            text = temp_file.read().decode('utf-8')
            # Split text by pages if possible
            # Note: pdfminer doesn't directly provide page breaks, so this is a workaround
            # Alternatively, use PyPDF2 for better page extraction
            # Here, we'll split by form feed character '\f'
            page_texts = text.split('\f')
            for i, page in enumerate(page_texts, start=1):
                clean_page = page.strip()
                if clean_page:
                    pages.append((i, clean_page))
        return pages
    except Exception as e:
        st.warning(f"Failed to extract text from PDF: {e}")
        return []

def extract_text_from_docx(file: io.BytesIO) -> List[Tuple[int, str]]:
    """
    Extracts text from a DOCX file on a per-paragraph basis.

    Args:
        file (io.BytesIO): The uploaded DOCX file.

    Returns:
        List[Tuple[int, str]]: A list of tuples containing paragraph numbers and extracted text.

    Example:
        paragraphs = extract_text_from_docx(uploaded_docx)
    """
    try:
        doc = Document(io.BytesIO(file.read()))
        paragraphs = []
        for i, para in enumerate(doc.paragraphs, start=1):
            clean_para = para.text.strip()
            if clean_para:
                paragraphs.append((i, clean_para))
        return paragraphs
    except Exception as e:
        st.warning(f"Failed to extract text from DOCX: {e}")
        return []

def extract_text_from_file(file) -> List[Dict]:
    """
    Determines the file type and extracts text accordingly, returning a list of chunks with metadata.

    Args:
        file (UploadedFile): The uploaded file from Streamlit.

    Returns:
        List[Dict]: A list of dictionaries containing chunk text and metadata.

    Example:
        chunks = extract_text_from_file(uploaded_file)
    """
    chunks = []
    if file.type == "application/pdf":
        pages = extract_text_from_pdf(file)
        for page_num, text in pages:
            page_chunks = chunk_text(text, max_length=500)
            for chunk in page_chunks:
                chunks.append({
                    "file_name": file.name,
                    "document_id": str(uuid.uuid4()),
                    "file_type": "pdf",
                    "page_number": page_num,
                    "chunk_text": chunk
                })
    elif file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]:
        paragraphs = extract_text_from_docx(file)
        for para_num, text in paragraphs:
            para_chunks = chunk_text(text, max_length=500)
            for chunk in para_chunks:
                chunks.append({
                    "file_name": file.name,
                    "document_id": str(uuid.uuid4()),
                    "file_type": "docx",
                    "paragraph_number": para_num,
                    "chunk_text": chunk
                })
    else:
        st.warning(f"Unsupported file type: {file.type}")
    return chunks

def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """
    Splits text into chunks based on periods, ensuring each chunk does not exceed max_length.

    Args:
        text (str): The input text to split.
        max_length (int): Maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.

    Example:
        chunks = chunk_text("Your long text here.", max_length=500)
    """
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += f". {sentence}"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = f"{sentence}"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embedding(text: str) -> list:
    """
    Generates an embedding vector for the given text using SentenceTransformer.

    Args:
        text (str): The input text to generate embedding for.

    Returns:
        list: The embedding vector.

    Example:
        embedding = generate_embedding("Sample text for embedding.")
    """
    try:
        embedding = embedding_model.encode(text)
        # Normalize the embedding to unit vector for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    except Exception as e:
        st.warning(f"Failed to generate embedding: {e}")
        return None

def store_document(chunk: Dict):
    """
    Stores a chunk's embedding and metadata in Qdrant.

    Args:
        chunk (Dict): A dictionary containing chunk text and metadata.

    Returns:
        None

    Example:
        store_document({"file_name": "doc.pdf", "document_id": "1234", "page_number": 1, "chunk_text": "Sample text."})
    """
    embedding = generate_embedding(chunk["chunk_text"])
    if embedding is None:
        st.sidebar.error(f"Failed to generate embedding for chunk in '{chunk['file_name']}'.")
        return
    payload = {
        "file_name": chunk["file_name"],
        "document_id": chunk["document_id"],
        "file_type": chunk["file_type"],
    }
    if chunk["file_type"] == "pdf":
        payload["page_number"] = chunk.get("page_number", "N/A")
    elif chunk["file_type"] == "docx":
        payload["paragraph_number"] = chunk.get("paragraph_number", "N/A")
    payload["chunk_text"] = chunk["chunk_text"]

    point = PointStruct(
        id=chunk["document_id"] + "_" + str(uuid.uuid4()),  # Unique ID per chunk
        vector=embedding,
        payload=payload
    )
    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        st.sidebar.success(f"Indexed a chunk from '{chunk['file_name']}'.")
    except Exception as e:
        st.sidebar.error(f"Failed to store chunk in Qdrant: {e}")

def query_similar_documents(query_embedding: list, top_k: int = 5):
    """
    Queries Qdrant for documents similar to the provided embedding.

    Args:
        query_embedding (list): The embedding vector of the query.
        top_k (int): Number of top similar documents to retrieve.

    Returns:
        list: A list of similar documents with their metadata and scores.

    Example:
        results = query_similar_documents([0.1, 0.2, ...], top_k=3)
    """
    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
        )
        return search_result
    except Exception as e:
        st.error(f"Failed to query Qdrant: {e}")
        return []

def get_file_icon(file_type: str) -> str:
    """
    Returns an emoji icon based on the file type.

    Args:
        file_type (str): The type of the file (e.g., 'pdf', 'docx').

    Returns:
        str: An emoji representing the file type.

    Example:
        icon = get_file_icon("pdf")  # Returns 📄
    """
    icons = {
        "pdf": "📄",
        "docx": "📃",
        "default": "📄"
    }
    return icons.get(file_type, icons["default"])

# =========================
# Streamlit Application
# =========================

def main():
    """
    The main function that defines the Streamlit app layout and interactions.

    Args:
        None

    Returns:
        None

    Example:
        main()
    """
    # Initialize Qdrant collection
    initialize_qdrant_collection()

    # App Title
    st.title("📄 Retrieval-Augmented Generation (RAG) App with Qdrant Cloud")

    # =========================
    # Sidebar: File Upload
    # =========================
    st.sidebar.header("📂 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.sidebar.write(f"Processing: {uploaded_file.name}")
            # Extract chunks from the uploaded file
            chunks = extract_text_from_file(uploaded_file)
            if not chunks:
                st.sidebar.warning(f"No text extracted from {uploaded_file.name}. Skipping.")
                continue

            # Store each chunk in Qdrant
            for chunk in chunks:
                store_document(chunk)

    st.sidebar.markdown("---")
    st.sidebar.info("Upload PDF or DOCX files to index them for searching.")

    # =========================
    # Main Section: Querying
    # =========================
    st.header("🔍 Search Documents")

    # Input field for user query
    user_query = st.text_input("Enter your query here:", "")

    # Search button
    if st.button("Search"):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Generating embedding and searching for similar documents..."):
                # Generate embedding for the query
                query_embedding = generate_embedding(user_query)
                if query_embedding is None:
                    st.error("Failed to generate embedding for the query.")
                else:
                    # Query Qdrant for similar documents
                    results = query_similar_documents(query_embedding, top_k=5)

                    if not results:
                        st.info("No similar documents found.")
                    else:
                        st.success(f"Found {len(results)} similar document(s):")
                        for idx, result in enumerate(results, start=1):
                            payload = result.payload
                            file_name = payload.get("file_name", "Unknown")
                            document_id = payload.get("document_id", "N/A")
                            file_type = payload.get("file_type", "default")
                            chunk_text = payload.get("chunk_text", "")
                            page_number = payload.get("page_number", None)
                            paragraph_number = payload.get("paragraph_number", None)
                            score = result.score

                            icon = get_file_icon(file_type)

                            # Display result with icon and metadata
                            with st.container():
                                col1, col2 = st.columns([1, 4])
                                with col1:
                                    st.markdown(icon)
                                with col2:
                                    st.markdown(f"**{file_name}**")
                                    st.markdown(f"**Document ID**: {document_id}")
                                    if file_type == "pdf" and page_number:
                                        st.markdown(f"**Page Number**: {page_number}")
                                    if file_type == "docx" and paragraph_number:
                                        st.markdown(f"**Paragraph Number**: {paragraph_number}")
                                    st.markdown(f"**Similarity Score**: {score:.4f}")
                                    st.markdown(f"**Snippet**: {chunk_text[:200]}...")  # Display first 200 chars
                                st.markdown("---")

    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and Qdrant Cloud.")

if __name__ == "__main__":
    main()