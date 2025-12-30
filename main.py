from fastapi import FastAPI
from pydantic import BaseModel
from services.parser import extract_all_resumes
from services.contact_extractor import extract_contact, extract_name
from services.embeddings import create_vector_store, create_tuned_retriever
from services.chunking import chunk_resumes
from services.skills_experience import extract_skills_experience
from langgraph_workflows.resume_analysis_workflow import analyze_resume_parallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import config
from typing import Optional

load_dotenv()

if config.API_KEY:
    os.environ["OPENAI_API_KEY"] = config.API_KEY

resume_folder = "data/resumes"
if not os.path.exists(resume_folder):
    os.makedirs(resume_folder, exist_ok=True)

# Extract resumes
resumes = extract_all_resumes(resume_folder)
resume_text_to_filename = {}

# RAG Configuration
CHUNK_SIZE = 3000  # ~750 tokens (adjustable: 2000-4000 for 500-1000 tokens)
CHUNK_OVERLAP = 400
TOP_K_CHUNKS = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score (0.0 to 1.0)
USE_LANGGRAPH_WORKFLOW = True  # Toggle between LangGraph workflow and sequential processing

if not resumes:
    resume_texts = []
    vector_store = None
    qa = None
    embeddings_model = None
    chunked_documents = []
else:
    resume_texts = list(resumes.values())
    resume_names = list(resumes.keys())
    for filename, text in resumes.items():
        resume_text_to_filename[text] = filename
    
    # Chunk resumes for better RAG retrieval
    chunked_documents = chunk_resumes(
        resumes,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Create vector store from chunked documents
    vector_store = create_vector_store(chunked_documents=chunked_documents)
    embeddings_model = vector_store.embeddings
    
    llm = ChatOpenAI(
        temperature=0,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/malikmoaz01/RecruitGenie",
            "X-Title": "RecruitGenie"
        }
    )
    
    # Create tuned retriever with configurable parameters
    retriever = create_tuned_retriever(
        vector_store,
        top_k=TOP_K_CHUNKS,
        search_type="similarity",
        similarity_threshold=SIMILARITY_THRESHOLD
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on the provided context from resume chunks.
        
Context: {context}

Question: {question}

Answer:"""
    )
    
    def format_docs(docs):
        # Group chunks by source filename for better context
        chunks_by_source = {}
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(doc.page_content)
        
        formatted = []
        for source, chunks in chunks_by_source.items():
            formatted.append(f"[From {source}]:\n" + "\n\n".join(chunks))
        
        return "\n\n---\n\n".join(formatted)
    
    qa = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

app = FastAPI(
    title="ResumeGenie API",
    description="AI-powered resume matching & contact extraction system",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Welcome to ResumeGenie API",
        "endpoints": {
            "docs": "/docs",
            "match_resumes": "/match-resumes (POST)"
        },
        "status": "running"
    }

class JobRequest(BaseModel):
    job_description: str
    top_n: int = 5
    use_langgraph: Optional[bool] = None  # Override default workflow setting
    chunk_size: Optional[int] = None  # Override default chunk size
    top_k_chunks: Optional[int] = None  # Override default top_k
    similarity_threshold: Optional[float] = None  # Override default threshold

@app.post("/match-resumes")
def match_resumes(request: JobRequest):
    query = request.job_description
    top_n = max(1, min(request.top_n, 10))
    use_workflow = request.use_langgraph if request.use_langgraph is not None else USE_LANGGRAPH_WORKFLOW
    
    if not resumes:
        return {"error": "No resumes found in data/resumes folder", "matches": []}
    
    if vector_store is None or qa is None or embeddings_model is None:
        return {"error": "Vector store or RAG chain not initialized", "matches": []}
    
    # Get RAG result from chunked documents
    result_text = qa.invoke(query)
    
    # Calculate similarity scores using chunk-based retrieval
    query_embedding = embeddings_model.embed_query(query)
    
    # Retrieve top chunks for similarity scoring
    top_k = request.top_k_chunks if request.top_k_chunks else TOP_K_CHUNKS
    threshold = request.similarity_threshold if request.similarity_threshold else SIMILARITY_THRESHOLD
    
    # Get relevant chunks from retriever (already filtered by similarity threshold if configured)
    relevant_docs = retriever.invoke(query)
    
    # Aggregate similarity scores by resume filename
    # Use maximum chunk similarity per resume for scoring
    resume_scores = {}
    
    for doc in relevant_docs:
        filename = doc.metadata.get("source", doc.metadata.get("filename", "unknown"))
        # Calculate similarity for this chunk
        chunk_embedding = embeddings_model.embed_query(doc.page_content)
        
        dot_product = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
        query_norm = sum(a * a for a in query_embedding) ** 0.5
        chunk_norm = sum(a * a for a in chunk_embedding) ** 0.5
        
        if query_norm > 0 and chunk_norm > 0:
            chunk_similarity = dot_product / (query_norm * chunk_norm)
            
            # Use maximum similarity score (best matching chunk represents the resume)
            if filename not in resume_scores or chunk_similarity > resume_scores[filename]:
                resume_scores[filename] = chunk_similarity
    
    # For resumes not in retrieved chunks, calculate similarity from full text
    for filename, text in resumes.items():
        if filename not in resume_scores:
            text_embedding = embeddings_model.embed_query(text[:2000])
            dot_product = sum(a * b for a, b in zip(query_embedding, text_embedding))
            query_norm = sum(a * a for a in query_embedding) ** 0.5
            text_norm = sum(a * a for a in text_embedding) ** 0.5
            
            if query_norm > 0 and text_norm > 0:
                cosine_similarity = dot_product / (query_norm * text_norm)
                resume_scores[filename] = cosine_similarity
    
    # Filter by similarity threshold and sort
    filtered_scores = {
        filename: score 
        for filename, score in resume_scores.items() 
        if score >= threshold
    }
    
    if not filtered_scores:
        # If no scores meet threshold, use all scores
        filtered_scores = resume_scores
    
    sorted_resumes = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
    top_resumes = sorted_resumes[:top_n]
    
    # Process top resumes using LangGraph workflow or sequential processing
    matches = []
    for filename, similarity_score in top_resumes:
        text = resumes[filename]
        
        if use_workflow:
            # Use LangGraph multi-agent workflow for parallel processing
            result = analyze_resume_parallel(
                resume_text=text,
                filename=filename,
                job_description=query,
                embeddings_model=embeddings_model,
                llm=llm
            )
            result["similarity_score"] = round(similarity_score, 4)
            matches.append(result)
        else:
            # Sequential processing (original approach)
            contact = extract_contact(text)
            extracted_name = extract_name(text)
            person_name = extracted_name if extracted_name else filename.replace('.pdf', '').replace('.docx', '')
            
            skills_exp = extract_skills_experience(text, llm)
            
            matches.append({
                "name": person_name,
                "similarity_score": round(similarity_score, 4),
                "emails": contact["emails"],
                "phones": contact["phones"],
                "skills": skills_exp["skills"],
                "experience_years": skills_exp["experience_years"],
                "work_experience": skills_exp["work_experience"]
            })
    
    return {
        "rag_result": result_text,
        "total_resumes": len(resumes),
        "total_chunks": len(chunked_documents),
        "top_n": top_n,
        "rag_config": {
            "chunk_size": CHUNK_SIZE,
            "top_k_chunks": top_k,
            "similarity_threshold": threshold,
            "use_langgraph_workflow": use_workflow
        },
        "matches": matches
    }
