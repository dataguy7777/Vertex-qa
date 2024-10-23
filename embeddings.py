# embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
from logger import logger
import streamlit as st

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # A lightweight, efficient model

@st.cache_resource
def load_embedding_model():
    """
    Loads the pre-trained SentenceTransformer model for generating embeddings.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        logger.error(f"Failed to load embedding model: {e}")
        st.stop()

embedding_model = load_embedding_model()

def generate_embedding(text: str) -> list:
    """
    Generates an embedding vector for the given text using SentenceTransformer.

    Args:
        text (str): The input text to generate embedding for.

    Returns:
        list: The embedding vector.
    """
    try:
        embedding = embedding_model.encode(text)
        # Normalize the embedding to unit vector for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        logger.debug("Generated and normalized embedding.")
        return embedding.tolist()
    except Exception as e:
        st.warning(f"Failed to generate embedding: {e}")
        logger.error(f"Failed to generate embedding: {e}")
        return None
