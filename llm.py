# llm.py

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from logger import logger
import streamlit as st

@st.cache_resource
def load_llm_pipeline():
    """
    Loads the pre-trained open-source LLM pipeline for answer generation.

    Returns:
        transformers.Pipeline: The loaded LLM pipeline.
    """
    try:
        model_name = "google/flan-t5-base"  # Small and efficient model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        logger.info(f"Loaded LLM model: {model_name}")
        return llm_pipeline
    except Exception as e:
        st.error(f"Failed to load LLM model: {e}")
        logger.error(f"Failed to load LLM model: {e}")
        st.stop()

llm_pipeline = load_llm_pipeline()

def generate_answer(query: str, context_chunks: List[str]) -> str:
    """
    Generates an answer to the user's query based on the context chunks using an open-source LLM.

    Args:
        query (str): The user's question.
        context_chunks (List[str]): List of text chunks from retrieved documents.

    Returns:
        str: The generated answer.
    """
    try:
        # Combine context chunks into a single context string
        context = "\n\n".join(context_chunks)
        # Structure the prompt as per the user's requirement
        prompt = f"The info found in documents collected explain that:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
        # Use the open-source LLM pipeline to generate an answer
        response = llm_pipeline(prompt, max_length=200, num_return_sequences=1)
        answer = response[0]['generated_text'].strip()
        logger.info("Generated answer using open-source LLM.")
        return answer
    except Exception as e:
        st.error(f"Failed to generate answer: {e}")
        logger.error(f"Failed to generate answer: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."
