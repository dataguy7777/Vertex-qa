# app.py

import os
import streamlit as st
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
INDEX_DISPLAY_NAME = os.getenv('INDEX_DISPLAY_NAME')
ENDPOINT_DISPLAY_NAME = os.getenv('ENDPOINT_DISPLAY_NAME')
DEPLOYED_INDEX_ID = os.getenv('DEPLOYED_INDEX_ID')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# Authenticate and initialize Vertex AI SDK
credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

def extract_text_from_word(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_ppt(file):
    prs = Presentation(file)
    full_text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                full_text.append(shape.text)
    return '\n'.join(full_text)

def extract_text_from_excel(file):
    workbook = load_workbook(filename=file, data_only=True)
    full_text = []
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        for row in sheet.iter_rows(values_only=True):
            row_text = ' '.join([str(cell) for cell in row if cell is not None])
            if row_text:
                full_text.append(row_text)
    return '\n'.join(full_text)

def preprocess_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.docx'):
            text = extract_text_from_word(uploaded_file)
        elif uploaded_file.name.endswith('.pptx'):
            text = extract_text_from_ppt(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            text = extract_text_from_excel(uploaded_file)
        else:
            continue  # Skip unsupported file types
        documents.append({
            'text': text,
            'metadata': {
                'file_name': uploaded_file.name
            }
        })
    return documents

def get_embeddings(texts):
    embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    embeddings = []
    for text in texts:
        result = embedding_model.get_embeddings([text])
        embeddings.append(result.embeddings[0].values)
    return embeddings

def create_index(embeddings, documents):
    # Create the Matching Engine Index
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents=embeddings,
        dimensions=len(embeddings[0]),
        approximate_neighbors_count=100,
        distance_measure_type='COSINE_DISTANCE',
    )
    index.wait()

    # Create Index Endpoint and deploy the index
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
    )
    index_endpoint.wait()

    deployed_index = index_endpoint.deploy_index(
        index=index,
        deployed_index_id=DEPLOYED_INDEX_ID,
    )
    deployed_index.wait()

    return index_endpoint

def query_index(index_endpoint, query_text):
    embedding_model = aiplatform.TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    query_embedding = embedding_model.get_embeddings([query_text]).embeddings[0].values

    # Perform nearest neighbor search
    response = index_endpoint.match(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embedding],
        num_neighbors=5,
    )

    # Extract matched documents
    matched_docs = []
    for neighbor in response[0].neighbors:
        # Assuming the IDs correspond to the order of uploaded documents
        idx = int(neighbor.datapoint_id)
        matched_docs.append(idx)
    return matched_docs

def generate_response(query_text, context_texts):
    # Prepare the prompt
    prompt = f"""Answer the following question based on the provided context.

Context:
{context_texts}

Question:
{query_text}

Answer:"""

    # Use Vertex AI language model
    llm = aiplatform.TextGenerationModel.from_pretrained("text-bison@001")
    response = llm.predict(prompt)
    return response.text

# Streamlit App

st.title("Retrieval Augmented Generation with Vertex AI")

# Sidebar for uploading documents
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload .docx, .pptx, .xlsx files",
    type=['docx', 'pptx', 'xlsx'],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            # Step 1: Preprocess documents
            documents = preprocess_documents(uploaded_files)

            # Step 2: Generate embeddings
            texts = [doc['text'] for doc in documents]
            embeddings = get_embeddings(texts)

            # Step 3: Create index and deploy endpoint
            index_endpoint = create_index(embeddings, documents)

            st.success("Documents processed and index created successfully!")
    else:
        st.warning("Please upload at least one document.")

# Main interface for querying
query_text = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if 'index_endpoint' in locals():
        with st.spinner("Retrieving information and generating answer..."):
            # Step 4: Handle user query
            matched_indices = query_index(index_endpoint, query_text)

            # Retrieve matched documents' text
            context_texts = ''
            for idx in matched_indices:
                doc = documents[idx]
                context_texts += doc['text'] + '\n'

            # Step 5: Generate response
            answer = generate_response(query_text, context_texts)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please process documents first by uploading them in the sidebar.")