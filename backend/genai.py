import os
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# IMPORTANT: You need a Hugging Face Hub API Token for this to work.
# 1. Go to https://huggingface.co/settings/tokens to get a token.
# 2. Set it as an environment variable BEFORE you run your FastAPI server.
#    In your terminal: set HUGGINGFACEHUB_API_TOKEN=hf_YourTokenHere
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    print("WARNING: Hugging Face API Token not found. GenAI features will fail.")

def get_basic_qna_chain():
    """Initializes a simple Q&A chain with a foundation model."""
    try:
        # Using a smaller, free model from Google that works well for simple Q&A
        repo_id = "google/flan-t5-large"
        llm = HuggingFaceHub(
            repo_id=repo_id, 
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        
        template = "Question: {question}\\n\\nAnswer: Let's think step by step."
        prompt = PromptTemplate(template=template, input_variables=["question"])
        
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None