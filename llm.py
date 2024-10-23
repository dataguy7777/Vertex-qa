# llm.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
from logger import logger
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.llms import HuggingFacePipeline
from langchain.callbacks import get_openai_callback

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

def get_llm_chain():
    """
    Sets up the LangChain LLM pipeline with GPT-J-6B.
    """
    try:
        pipe = torch.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if DEVICE == "cuda" else -1,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Initialized LangChain LLM pipeline.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM pipeline: {e}")
        logger.error(f"Failed to initialize LLM pipeline: {e}")
        st.stop()

llm = get_llm_chain()

# Define the prompt template
prompt_template = """
The information found in the documents collected explains that:

{context}

Question: {question}

Answer:
"""

# Initialize the PromptTemplate
template = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Initialize the LLM Chain
llm_chain = LLMChain(llm=llm, prompt=template)

# Initialize the agent with the LLM chain
agent = initialize_agent(
    tools=[],  # No additional tools for now
    llm=llm_chain,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

def generate_answer(query: str, context_chunks: List[str]) -> str:
    """
    Generates an answer to the user's query based on the context chunks using an agent.

    Args:
        query (str): The user's question.
        context_chunks (List[str]): List of text chunks from retrieved documents.

    Returns:
        str: The generated answer.
    """
    try:
        # Combine context chunks into a single context string
        context = "\n\n".join(context_chunks)
        
        # Use the agent to generate the answer
        with get_openai_callback() as cb:
            response = agent.run({"context": context, "question": query})
        
        # Format the answer
        answer = f"📝 **Answer**\n\n{response}"
        logger.info("Generated answer using LangChain agent with GPT-J-6B.")
        return answer
    except Exception as e:
        st.error(f"Failed to generate answer: {e}")
        logger.error(f"Failed to generate answer: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."
