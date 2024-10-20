# pages/2_Vector_Visualization.py

import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import logging
import io
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd

# =========================
# Logging Configuration
# =========================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# Load Environment Variables
# =========================

load_dotenv()

# Retrieve environment variables
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g., "https://your-instance.qdrant.io"

# Validate environment variables
if not all([QDRANT_API_KEY, QDRANT_URL]):
    st.error("Please ensure that QDRANT_API_KEY and QDRANT_URL are set in the .env file.")
    logger.error("Missing QDRANT_API_KEY or QDRANT_URL in .env file.")
    st.stop()

# =========================
# Initialize Qdrant Client
# =========================

try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    logger.info("Successfully connected to Qdrant Cloud.")
except Exception as e:
    st.error(f"Failed to connect to Qdrant Cloud: {e}")
    logger.error(f"Failed to connect to Qdrant Cloud: {e}")
    st.stop()

# Define Qdrant collection name
COLLECTION_NAME = "documents"

# =========================
# Initialize Sentence Transformer Model
# =========================

@st.cache_resource
def load_embedding_model():
    """
    Loads the pre-trained SentenceTransformer model for generating embeddings.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        logger.error(f"Failed to load embedding model: {e}")
        st.stop()

embedding_model = load_embedding_model()

# =========================
# Utility Functions
# =========================

def fetch_all_vectors():
    """
    Fetches all vectors and their metadata from the Qdrant collection.

    Returns:
        np.ndarray: Array of embedding vectors.
        list: List of metadata dictionaries corresponding to each vector.
    """
    try:
        vectors = []
        metadata = []
        # Qdrant's search may have limits; consider using scroll or pagination if dataset is large
        # For simplicity, assuming a manageable number of vectors
        response = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000  # Adjust based on your dataset size
        )
        vectors.extend([point.vector for point in response.points])
        metadata.extend([point.payload for point in response.points])

        while response.next_page_offset:
            response = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=response.next_page_offset
            )
            vectors.extend([point.vector for point in response.points])
            metadata.extend([point.payload for point in response.points])

        logger.info(f"Fetched {len(vectors)} vectors from Qdrant.")
        return np.array(vectors), metadata
    except Exception as e:
        st.error(f"Failed to fetch vectors from Qdrant: {e}")
        logger.error(f"Failed to fetch vectors from Qdrant: {e}")
        return np.array([]), []

def reduce_dimensionality(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduces the dimensionality of embedding vectors using PCA.

    Args:
        vectors (np.ndarray): High-dimensional embedding vectors.
        n_components (int): Number of dimensions to reduce to.

    Returns:
        np.ndarray: 2D embedding vectors.
    """
    try:
        pca = PCA(n_components=n_components, random_state=42)
        reduced_vectors = pca.fit_transform(vectors)
        logger.info(f"Reduced dimensionality from {vectors.shape[1]} to {n_components}.")
        return reduced_vectors
    except Exception as e:
        st.error(f"Failed to reduce dimensionality: {e}")
        logger.error(f"Failed to reduce dimensionality: {e}")
        return np.array([])

def create_interactive_plot(vectors_2d: np.ndarray, metadata: list) -> pd.DataFrame:
    """
    Creates an interactive Plotly scatter plot of the 2D vectors.

    Args:
        vectors_2d (np.ndarray): 2D embedding vectors.
        metadata (list): List of metadata dictionaries corresponding to each vector.

    Returns:
        pd.DataFrame: DataFrame containing vector coordinates and metadata.
    """
    try:
        # Prepare DataFrame for Plotly
        df = pd.DataFrame({
            'x': vectors_2d[:, 0],
            'y': vectors_2d[:, 1],
            'file_name': [md.get('file_name', 'Unknown') for md in metadata],
            'document_id': [md.get('document_id', 'N/A') for md in metadata],
            'file_type': [md.get('file_type', 'default') for md in metadata],
            'chunk_text': [md.get('chunk_text', '') for md in metadata],
            'page_number': [md.get('page_number', None) for md in metadata],
            'paragraph_number': [md.get('paragraph_number', None) for md in metadata]
        })

        # Create Plotly scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            hover_data=['file_name', 'document_id', 'file_type'],
            title='2D Visualization of Document Embeddings',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
            width=800,
            height=600
        )

        # Update marker size and color
        fig.update_traces(marker=dict(size=8, color='blue'), selector=dict(mode='markers'))

        # Display Plotly chart and capture click events
        selected_points = st.plotly_chart(fig, use_container_width=True)

        # Use Streamlit's Plotly event to capture clicks
        clicked_point = None
        if 'plotly_click' in st.session_state:
            clicked_point = st.session_state['plotly_click']
            logger.info(f"Point clicked: {clicked_point}")

        # Display selected point details
        if clicked_point:
            point_index = clicked_point['pointIndex']
            selected_metadata = metadata[point_index]
            st.sidebar.markdown("---")
            st.sidebar.subheader("📄 Document Details")
            st.sidebar.markdown(f"**File Name**: {selected_metadata.get('file_name', 'Unknown')}")
            st.sidebar.markdown(f"**Document ID**: {selected_metadata.get('document_id', 'N/A')}")
            st.sidebar.markdown(f"**File Type**: {selected_metadata.get('file_type', 'default').upper()}")
            if selected_metadata.get('file_type') == 'pdf':
                st.sidebar.markdown(f"**Page Number**: {selected_metadata.get('page_number', 'N/A')}")
            if selected_metadata.get('file_type') == 'docx':
                st.sidebar.markdown(f"**Paragraph Number**: {selected_metadata.get('paragraph_number', 'N/A')}")
            st.sidebar.markdown(f"**Snippet**: {selected_metadata.get('chunk_text', '')[:500]}...")
            st.sidebar.markdown("---")

        return df
    except Exception as e:
        st.error(f"Failed to create interactive plot: {e}")
        logger.error(f"Failed to create interactive plot: {e}")
        return pd.DataFrame()

def get_file_icon(file_type: str) -> str:
    """
    Returns an emoji icon based on the file type.

    Args:
        file_type (str): The type of the file (e.g., 'pdf', 'docx').

    Returns:
        str: An emoji representing the file type.
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
    The main function that defines the Streamlit app layout and interactions for Vector Visualization.

    Args:
        None

    Returns:
        None

    Example:
        main()
    """
    st.title("📈 Vector Visualization in 2D Space")

    # Fetch all vectors and metadata
    with st.spinner("Fetching vectors from Qdrant..."):
        vectors, metadata = fetch_all_vectors()

    if vectors.size == 0:
        st.error("No vectors available for visualization.")
        logger.error("No vectors retrieved from Qdrant.")
        st.stop()

    # Dimensionality Reduction
    with st.spinner("Reducing dimensionality with PCA..."):
        vectors_2d = reduce_dimensionality(vectors, n_components=2)

    if vectors_2d.size == 0:
        st.error("Failed to reduce dimensionality.")
        logger.error("Dimensionality reduction resulted in empty vectors.")
        st.stop()

    # Create Interactive Plot
    with st.spinner("Creating interactive plot..."):
        df = create_interactive_plot(vectors_2d, metadata)

    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit, Qdrant Cloud, and Plotly.")

if __name__ == "__main__":
    main()