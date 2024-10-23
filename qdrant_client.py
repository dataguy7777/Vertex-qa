# qdrant_client.py

import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from logger import logger
import streamlit as st
from utils import chunk_text

COLLECTION_NAME = "documents"
VECTOR_SIZE = 384  # Dimension size for all-MiniLM-L6-v2

# Initialize Qdrant Client
def initialize_qdrant_collection():
    """
    Initializes the Qdrant collection with the specified schema.
    If the collection already exists, it will not be recreated.
    """
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]

    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
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
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' created successfully.")
        else:
            st.sidebar.info(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
        return qdrant_client
    except Exception as e:
        st.sidebar.error(f"Error initializing Qdrant collection: {e}")
        logger.error(f"Error initializing Qdrant collection: {e}")
        st.stop()

# Store a single document chunk in Qdrant
def store_document(qdrant_client, chunk: Dict):
    """
    Stores a chunk's embedding and metadata in Qdrant.

    Args:
        qdrant_client (QdrantClient): The initialized Qdrant client.
        chunk (Dict): A dictionary containing chunk text and metadata.

    Returns:
        None
    """
    from embeddings import generate_embedding

    embedding = generate_embedding(chunk["chunk_text"])
    if embedding is None:
        st.sidebar.error(f"Failed to generate embedding for chunk in '{chunk['file_name']}'.")
        logger.error(f"Failed to generate embedding for chunk in '{chunk['file_name']}'.")
        return

    payload = {
        "file_name": chunk["file_name"],
        "document_id": chunk["document_id"],
        "file_type": chunk["file_type"],
    }

    # Add specific metadata based on file type
    if chunk["file_type"] == "pdf":
        payload["page_number"] = chunk.get("page_number", "N/A")
    elif chunk["file_type"] == "docx":
        payload["paragraph_number"] = chunk.get("paragraph_number", "N/A")
    elif chunk["file_type"] == "pptx":
        payload["slide_number"] = chunk.get("slide_number", "N/A")
    elif chunk["file_type"] == "xlsx":
        payload["cell_id"] = chunk.get("cell_id", "N/A")
    elif chunk["file_type"] == "txt":
        payload["line_number"] = chunk.get("line_number", "N/A")

    # Optionally, add a thumbnail if available
    # payload["thumbnail"] = chunk.get("thumbnail", None)

    # Use a single UUID for point ID
    point_id = str(uuid.uuid4())

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload=payload
    )
    try:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        st.sidebar.success(f"Indexed a chunk from '{chunk['file_name']}'.")
        logger.info(f"Indexed chunk ID: {point_id} from '{chunk['file_name']}'.")
    except Exception as e:
        st.sidebar.error(f"Failed to store chunk in Qdrant: {e}")
        logger.error(f"Failed to store chunk ID: {point_id} in Qdrant: {e}")

# Query similar documents from Qdrant
def query_similar_documents(qdrant_client, query_embedding: list, top_k: int = 5):
    """
    Queries Qdrant for documents similar to the provided embedding.

    Args:
        qdrant_client (QdrantClient): The initialized Qdrant client.
        query_embedding (list): The embedding vector of the query.
        top_k (int): Number of top similar documents to retrieve.

    Returns:
        list: A list of similar documents with their metadata and scores.
    """
    try:
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
        )
        logger.info(f"Queried Qdrant for top {top_k} similar documents.")
        return search_result
    except Exception as e:
        st.error(f"Failed to query Qdrant: {e}")
        logger.error(f"Failed to query Qdrant: {e}")
        return []
