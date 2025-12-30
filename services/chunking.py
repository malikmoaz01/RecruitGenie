"""
Resume Chunking Service

Splits large resumes into smaller chunks for better RAG retrieval.
Uses character-based chunking (approximately 2000-4000 chars ≈ 500-1000 tokens).
"""
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str]
) -> List[str]:
    """
    Recursively split text by separators to create chunks.
    
    Args:
        text: Text to split
        chunk_size: Target chunk size in characters
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to try (in order of preference)
    
    Returns:
        List of text chunks
    """
    # If text is small enough, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # Calculate end position
        end_pos = min(current_pos + chunk_size, len(text))
        
        # Try to find a good break point using separators
        chunk_text = text[current_pos:end_pos]
        
        if end_pos < len(text):
            # Try to find a separator near the end for a clean break
            best_split_pos = end_pos
            for separator in separators:
                # Look for separator in the last 20% of chunk
                search_start = max(0, end_pos - int(chunk_size * 0.2))
                last_sep_pos = chunk_text.rfind(separator)
                if last_sep_pos > search_start - current_pos:
                    best_split_pos = current_pos + last_sep_pos + len(separator)
                    break
            
            chunk_text = text[current_pos:best_split_pos].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Move position with overlap
            current_pos = best_split_pos - chunk_overlap if best_split_pos > chunk_overlap else best_split_pos
        else:
            # Last chunk
            chunk_text = text[current_pos:].strip()
            if chunk_text:
                chunks.append(chunk_text)
            break
    
    return chunks if chunks else [text]


def chunk_resumes(
    resumes: Dict[str, str],
    chunk_size: int = 3000,
    chunk_overlap: int = 400
) -> List[Dict[str, Any]]:
    """
    Split resumes into chunks with metadata tracking.
    
    Args:
        resumes: Dictionary mapping filename to resume text
        chunk_size: Target chunk size in characters (3000 chars ≈ 750 tokens, 4000 chars ≈ 1000 tokens)
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of dictionaries containing chunk text and metadata (filename, chunk_index)
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunked_documents = []
    
    for filename, text in resumes.items():
        if not text.strip():
            continue
            
        chunks = _split_text_recursive(text, chunk_size, chunk_overlap, separators)
        
        for idx, chunk_text in enumerate(chunks):
            chunked_documents.append({
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "source": filename
                }
            })
    
    return chunked_documents


def chunk_single_resume(
    text: str,
    filename: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 400
) -> List[Dict[str, Any]]:
    """
    Split a single resume into chunks.
    
    Args:
        text: Resume text content
        filename: Source filename
        chunk_size: Target chunk size in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = _split_text_recursive(text, chunk_size, chunk_overlap, separators)
    chunked_documents = []
    
    for idx, chunk_text in enumerate(chunks):
        chunked_documents.append({
            "text": chunk_text,
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "source": filename
            }
        })
    
    return chunked_documents

