from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_vector_store(
    resume_texts: Optional[List[str]] = None,
    chunked_documents: Optional[List[Dict[str, Any]]] = None
):
    """
    Create a FAISS vector store from resume texts or chunked documents.
    
    Args:
        resume_texts: List of full resume texts (legacy support)
        chunked_documents: List of dictionaries with 'text' and 'metadata' keys
    
    Returns:
        FAISS vector store with embeddings
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    
    if chunked_documents:
        # Create Document objects from chunked documents
        documents = [
            Document(
                page_content=doc["text"],
                metadata=doc.get("metadata", {})
            )
            for doc in chunked_documents
        ]
        vector_store = FAISS.from_documents(documents, embeddings)
    elif resume_texts:
        # Legacy support for full resume texts
        vector_store = FAISS.from_texts(resume_texts, embeddings)
    else:
        raise ValueError("Either resume_texts or chunked_documents must be provided")
    
    return vector_store


def create_tuned_retriever(
    vector_store,
    top_k: int = 5,
    search_type: str = "similarity",
    similarity_threshold: Optional[float] = None
):
    """
    Create a retriever with tuned parameters.
    
    Args:
        vector_store: FAISS vector store
        top_k: Number of top chunks to retrieve
        search_type: Type of search ("similarity" or "mmr")
        similarity_threshold: Minimum similarity score threshold (0.0 to 1.0)
    
    Returns:
        Configured retriever
    """
    search_kwargs = {"k": top_k}
    
    if similarity_threshold is not None:
        search_kwargs["score_threshold"] = similarity_threshold
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    return retriever
