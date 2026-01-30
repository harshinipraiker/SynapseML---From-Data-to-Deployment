import pandas as pd
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_tabular_data_from_bytes(file_content: bytes, filename: str):
    """Loads tabular data from file content in bytes."""
    import io
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_content))
    else:
        raise ValueError("Unsupported file type for tabular data.")

def load_unstructured_documents(file_path: str):
    """Loads and splits documents for GenAI tasks (PDF, TXT)."""
    _, extension = os.path.splitext(file_path)
    
    if extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported document type: {extension}")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs