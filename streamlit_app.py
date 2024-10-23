# app.py

import streamlit as st
import uuid
from typing import List, Dict
from embeddings import load_embedding_model, generate_embedding
from qdrant_client import initialize_qdrant_collection, store_document, query_similar_documents
from llm import generate_answer
from utils import extract_text_from_file, get_file_icon, get_thumbnail
from logger import logger

# =========================
# Streamlit Page Configuration
# =========================

st.set_page_config(
    page_title="📄 RAG App with Qdrant Cloud",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Initialize Components
# =========================

# Initialize Qdrant collection
qdrant_client = initialize_qdrant_collection()

# App Title
st.title("📄 Retrieval-Augmented Generation (RAG) App with Qdrant Cloud")

# =========================
# Sidebar: File Upload
# =========================
st.sidebar.header("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Choose files", type=["pdf", "docx", "xlsx", "pptx", "txt"], accept_multiple_files=True
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
            store_document(qdrant_client, chunk)

st.sidebar.markdown("---")
st.sidebar.info("Upload PDF, DOCX, XLSX, PPTX, or TXT files to index them for searching.")

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
        logger.warning("User submitted an empty query.")
    else:
        with st.spinner("Generating embedding and searching for similar documents..."):
            # Generate embedding for the query
            query_embedding = generate_embedding(user_query)
            if query_embedding is None:
                st.error("Failed to generate embedding for the query.")
            else:
                # Query Qdrant for similar documents
                results = query_similar_documents(qdrant_client, query_embedding, top_k=5)

                if not results:
                    st.info("No similar documents found.")
                else:
                    st.success(f"Found {len(results)} similar document(s):")
                    context_chunks = []
                    for idx, result in enumerate(results, start=1):
                        payload = result.payload
                        file_name = payload.get("file_name", "Unknown")
                        document_id = payload.get("document_id", "N/A")
                        file_type = payload.get("file_type", "default")
                        chunk_text = payload.get("chunk_text", "")
                        page_number = payload.get("page_number", None)
                        paragraph_number = payload.get("paragraph_number", None)
                        slide_number = payload.get("slide_number", None)
                        cell_id = payload.get("cell_id", None)
                        line_number = payload.get("line_number", None)
                        score = result.score
                        thumbnail = payload.get("thumbnail", None)

                        icon = get_file_icon(file_type)

                        # Collect context chunks for answer generation
                        context_chunks.append(chunk_text)

                        # Display result with icon and metadata
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if thumbnail:
                                    # Display thumbnail
                                    st.image(thumbnail, width=50)
                                else:
                                    # Display icon
                                    st.markdown(icon)
                            with col2:
                                st.markdown(f"**{file_name}**")
                                st.markdown(f"**Document ID**: {document_id}")
                                if file_type == "pdf" and page_number:
                                    st.markdown(f"**Page Number**: {page_number}")
                                if file_type == "docx" and paragraph_number:
                                    st.markdown(f"**Paragraph Number**: {paragraph_number}")
                                if file_type == "pptx" and slide_number:
                                    st.markdown(f"**Slide Number**: {slide_number}")
                                if file_type == "xlsx" and cell_id:
                                    st.markdown(f"**Cell ID**: {cell_id}")
                                if file_type == "txt" and line_number:
                                    st.markdown(f"**Line Number**: {line_number}")
                                st.markdown(f"**Similarity Score**: {score:.4f}")
                                st.markdown(f"**Snippet**: {chunk_text[:200]}...")  # Display first 200 chars
                            st.markdown("---")

                    # Generate an answer based on the retrieved context
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(user_query, context_chunks)
                        st.header("📝 Answer")
                        st.write(answer)

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Qdrant Cloud.")
