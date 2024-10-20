# app.py

import streamlit as st
import os
import uuid
import io
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionConfig,
    Distance,
    VectorParams,
    PointStruct,
)
from pdfminer.high_level import extract_text
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# =========================
# Configuration and Setup
# =========================

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and URLs from environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g., "https://your-instance.qdrant.io"

# Validate that all necessary environment variables are set
if not all([QDRANT_API_KEY, QDRANT_URL]):
    st.error("Please ensure that QDRANT_API_KEY and QDRANT_URL are set in the .env file.")
    st.stop()

# Initialize Qdrant Client
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

# Initialize Sentence Transformer model
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
    If the collection already exists, it will be recreated.

    Example:
        initialize_qdrant_collection()

    Output:
        None
    """
    try:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        st.success(f"Qdrant collection '{COLLECTION_NAME}' is ready.")
    except Exception as e:
        st.error(f"Error initializing Qdrant collection: {e}")
        st.stop()

def extract_text_from_pdf(file: io.BytesIO) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file (io.BytesIO): The uploaded PDF file.

    Returns:
        str: Extracted text content.

    Example:
        text = extract_text_from_pdf(uploaded_pdf)
    """
    try:
        text = extract_text(file)
        return text
    except Exception as e:
        st.warning(f"Failed to extract text from PDF: {e}")
        return ""

def extract_text_from_docx(file: io.BytesIO) -> str:
    """
    Extracts text from a DOCX file.

    Args:
        file (io.BytesIO): The uploaded DOCX file.

    Returns:
        str: Extracted text content.

    Example:
        text = extract_text_from_docx(uploaded_docx)
    """
    try:
        doc = Document(io.BytesIO(file.read()))
        full_text = "\n".join([para.text for para in doc.paragraphs])
        return full_text
    except Exception as e:
        st.warning(f"Failed to extract text from DOCX: {e}")
        return ""

def extract_text_from_file(file) -> str:
    """
    Determines the file type and extracts text accordingly.

    Args:
        file (UploadedFile): The uploaded file from Streamlit.

    Returns:
        str: Extracted text content.

    Example:
        text = extract_text_from_file(uploaded_file)
    """
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]:
        return extract_text_from_docx(file)
    else:
        st.warning(f"Unsupported file type: {file.type}")
        return ""

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

def store_document(document_id: str, embedding: list, metadata: dict):
    """
    Stores a document's embedding and metadata in Qdrant.

    Args:
        document_id (str): Unique identifier for the document.
        embedding (list): The embedding vector of the document.
        metadata (dict): Metadata associated with the document.

    Returns:
        None

    Example:
        store_document("1234", [0.1, 0.2, ...], {"file_name": "doc.pdf"})
    """
    point = PointStruct(
        id=document_id,
        vector=embedding,
        payload=metadata
    )
    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        st.success(f"Document '{metadata.get('file_name')}' indexed successfully.")
    except Exception as e:
        st.error(f"Failed to store document in Qdrant: {e}")

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

    # Set Streamlit page configuration
    st.set_page_config(page_title="📄 RAG App with Qdrant Cloud", layout="wide")

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
            # Extract text from the uploaded file
            text = extract_text_from_file(uploaded_file)
            if not text:
                st.sidebar.warning(f"No text extracted from {uploaded_file.name}. Skipping.")
                continue

            # Generate a unique document ID
            document_id = str(uuid.uuid4())

            # Prepare metadata
            metadata = {
                "file_name": uploaded_file.name,
                "document_id": document_id,
                # Additional metadata can be added here
            }

            # Generate embedding for the extracted text
            embedding = generate_embedding(text)
            if embedding is None:
                st.sidebar.error(f"Failed to generate embedding for {uploaded_file.name}.")
                continue

            # Store the document in Qdrant
            store_document(document_id, embedding, metadata)

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
                            score = result.score
                            st.markdown(f"### {idx}. {file_name}")
                            st.markdown(f"**Document ID**: {document_id}")
                            st.markdown(f"**Similarity Score**: {score:.4f}")
                            st.markdown("---")

    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and Qdrant Cloud.")

if __name__ == "__main__":
    main()