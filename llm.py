# llm.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
from logger import logger
import streamlit as st

# Constants
MODEL_NAME = "EleutherAI/gpt-j-6B"  # More powerful open-source LLM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_llm():
    """
    Loads the GPT-J-6B model and tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        model.to(DEVICE)
        logger.info(f"Loaded LLM model: {MODEL_NAME} on {DEVICE}")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load LLM model: {e}")
        logger.error(f"Failed to load LLM model: {e}")
        st.stop()

model, tokenizer = load_llm()

def generate_answer(query: str, context_chunks: List[str]) -> str:
    """
    Generates an answer to the user's query based on the context chunks using GPT-J-6B.

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
        prompt = f"The information found in the documents collected explains that:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        # Generate response
        outputs = model.generate(
            inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Post-process the answer to match desired format
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        logger.info("Generated answer using GPT-J-6B.")
        return f"📝 **Answer**\n\n{answer}"
    except Exception as e:
        st.error(f"Failed to generate answer: {e}")
        logger.error(f"Failed to generate answer: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."
