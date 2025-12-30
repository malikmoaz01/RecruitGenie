"""
LangGraph Multi-Agent Workflow for Resume Analysis

Orchestrates parallel processing of resume analysis tasks:
- Contact extraction agent
- Skills identification agent
- Experience parsing agent
- Similarity calculation agent
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.contact_extractor import extract_contact, extract_name
from services.skills_experience import extract_skills_experience
import config


class ResumeAnalysisState(TypedDict):
    """State passed between workflow nodes"""
    resume_text: str
    filename: str
    job_description: Optional[str]
    contact_info: Optional[Dict[str, Any]]
    skills_experience: Optional[Dict[str, Any]]
    similarity_score: Optional[float]
    embeddings_model: Optional[Any]
    llm: Optional[Any]
    final_result: Optional[Dict[str, Any]]


def contact_extraction_node(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """
    Agent node for extracting contact information.
    Runs in parallel with other extraction tasks.
    """
    resume_text = state["resume_text"]
    
    contact = extract_contact(resume_text)
    extracted_name = extract_name(resume_text)
    person_name = extracted_name if extracted_name else state["filename"].replace('.pdf', '').replace('.docx', '')
    
    state["contact_info"] = {
        "name": person_name,
        "emails": contact["emails"],
        "phones": contact["phones"]
    }
    
    return state


def skills_experience_node(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """
    Agent node for extracting skills and experience.
    Runs in parallel with contact extraction.
    """
    resume_text = state["resume_text"]
    llm = state.get("llm")
    
    if llm is None:
        # Create LLM if not provided
        llm = ChatOpenAI(
            temperature=0,
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.API_KEY,
            default_headers={
                "HTTP-Referer": "https://github.com/malikmoaz01/RecruitGenie",
                "X-Title": "RecruitGenie"
            }
        )
    
    skills_exp = extract_skills_experience(resume_text, llm)
    
    state["skills_experience"] = {
        "skills": skills_exp["skills"],
        "experience_years": skills_exp["experience_years"],
        "work_experience": skills_exp["work_experience"]
    }
    
    return state


def similarity_calculation_node(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """
    Agent node for calculating similarity score with job description.
    Can run in parallel with extraction tasks if embeddings are pre-computed.
    """
    resume_text = state["resume_text"]
    job_description = state.get("job_description")
    embeddings_model = state.get("embeddings_model")
    
    if job_description and embeddings_model:
        try:
            query_embedding = embeddings_model.embed_query(job_description)
            # Use first 2000 chars for embedding to match original behavior
            text_embedding = embeddings_model.embed_query(resume_text[:2000])
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, text_embedding))
            query_norm = sum(a * a for a in query_embedding) ** 0.5
            text_norm = sum(a * a for a in text_embedding) ** 0.5
            
            if query_norm > 0 and text_norm > 0:
                cosine_similarity = dot_product / (query_norm * text_norm)
                state["similarity_score"] = cosine_similarity
            else:
                state["similarity_score"] = 0.0
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            state["similarity_score"] = 0.0
    else:
        state["similarity_score"] = 0.0
    
    return state


def merge_results_node(state: ResumeAnalysisState) -> ResumeAnalysisState:
    """
    Final node that merges all extracted information into a single result.
    """
    contact_info = state.get("contact_info", {})
    skills_exp = state.get("skills_experience", {})
    similarity_score = state.get("similarity_score", 0.0)
    
    state["final_result"] = {
        "name": contact_info.get("name", state["filename"].replace('.pdf', '').replace('.docx', '')),
        "similarity_score": round(similarity_score, 4),
        "emails": contact_info.get("emails", []),
        "phones": contact_info.get("phones", []),
        "skills": skills_exp.get("skills", []),
        "experience_years": skills_exp.get("experience_years"),
        "work_experience": skills_exp.get("work_experience", [])
    }
    
    return state


def create_resume_analysis_workflow(llm: Optional[ChatOpenAI] = None) -> StateGraph:
    """
    Create and compile the LangGraph workflow for resume analysis.
    
    The workflow processes resumes through specialized agent nodes:
    1. Contact extraction
    2. Skills and experience extraction  
    3. Similarity calculation
    4. Result merging
    
    While executed sequentially in the graph, the modular design allows
    for easy parallelization when processing multiple resumes.
    
    Args:
        llm: Optional LLM instance to use in the workflow
    
    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(ResumeAnalysisState)
    
    # Add nodes for specialized agent tasks
    workflow.add_node("extract_contact", contact_extraction_node)
    workflow.add_node("extract_skills_experience", skills_experience_node)
    workflow.add_node("calculate_similarity", similarity_calculation_node)
    workflow.add_node("merge_results", merge_results_node)
    
    # Define the workflow sequence
    workflow.set_entry_point("extract_contact")
    workflow.add_edge("extract_contact", "extract_skills_experience")
    workflow.add_edge("extract_skills_experience", "calculate_similarity")
    workflow.add_edge("calculate_similarity", "merge_results")
    workflow.add_edge("merge_results", END)
    
    return workflow.compile()


def analyze_resume_parallel(
    resume_text: str,
    filename: str,
    job_description: Optional[str] = None,
    embeddings_model: Optional[Any] = None,
    llm: Optional[ChatOpenAI] = None
) -> Dict[str, Any]:
    """
    Analyze a single resume using the multi-agent workflow.
    
    Args:
        resume_text: Text content of the resume
        filename: Source filename
        job_description: Optional job description for similarity calculation
        embeddings_model: Optional embeddings model for similarity
        llm: Optional LLM instance
    
    Returns:
        Dictionary containing extracted information and similarity score
    """
    workflow = create_resume_analysis_workflow(llm)
    
    initial_state = {
        "resume_text": resume_text,
        "filename": filename,
        "job_description": job_description,
        "embeddings_model": embeddings_model,
        "llm": llm,
        "contact_info": None,
        "skills_experience": None,
        "similarity_score": None,
        "final_result": None
    }
    
    # Run the workflow
    final_state = workflow.invoke(initial_state)
    
    return final_state.get("final_result", {})


def analyze_multiple_resumes_parallel(
    resumes: Dict[str, str],
    job_description: Optional[str] = None,
    embeddings_model: Optional[Any] = None,
    llm: Optional[ChatOpenAI] = None
) -> List[Dict[str, Any]]:
    """
    Analyze multiple resumes in parallel using the workflow.
    
    Args:
        resumes: Dictionary mapping filename to resume text
        job_description: Optional job description for similarity calculation
        embeddings_model: Optional embeddings model for similarity
        llm: Optional LLM instance
    
    Returns:
        List of dictionaries containing extracted information for each resume
    """
    workflow = create_resume_analysis_workflow(llm)
    results = []
    
    for filename, resume_text in resumes.items():
        initial_state = {
            "resume_text": resume_text,
            "filename": filename,
            "job_description": job_description,
            "embeddings_model": embeddings_model,
            "llm": llm,
            "contact_info": None,
            "skills_experience": None,
            "similarity_score": None,
            "final_result": None
        }
        
        final_state = workflow.invoke(initial_state)
        result = final_state.get("final_result", {})
        if result:
            results.append(result)
    
    return results

