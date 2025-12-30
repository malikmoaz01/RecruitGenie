from fastapi import FastAPI
from pydantic import BaseModel
from services.parser import extract_all_resumes
from services.contact_extractor import extract_contact, extract_name
from services.embeddings import create_vector_store
from services.skills_experience import extract_skills_experience
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import config

load_dotenv()

if config.API_KEY:
    os.environ["OPENAI_API_KEY"] = config.API_KEY

resume_folder = "data/resumes"
if not os.path.exists(resume_folder):
    os.makedirs(resume_folder, exist_ok=True)

resumes = extract_all_resumes(resume_folder)
resume_text_to_filename = {}
if not resumes:
    resume_texts = []
    vector_store = None
    qa = None
    embeddings_model = None
else:
    resume_texts = list(resumes.values())
    resume_names = list(resumes.keys())
    for filename, text in resumes.items():
        resume_text_to_filename[text] = filename
    vector_store = create_vector_store(resume_texts)
    embeddings_model = vector_store.embeddings
    
    llm = ChatOpenAI(
        temperature=0,
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.API_KEY,
        default_headers={
            "HTTP-Referer": "https://github.com/yourusername/RecruitGenie",
            "X-Title": "RecruitGenie"
        }
    )
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on the provided context from resumes.
        
Context: {context}

Question: {question}

Answer:"""
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
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

@app.post("/match-resumes")
def match_resumes(request: JobRequest):
    query = request.job_description
    top_n = max(1, min(request.top_n, 10))
    
    if not resumes:
        return {"error": "No resumes found in data/resumes folder", "matches": []}
    
    if vector_store is None or qa is None or embeddings_model is None:
        return {"error": "Vector store or RAG chain not initialized", "matches": []}
    
    result_text = qa.invoke(query)
    
    query_embedding = embeddings_model.embed_query(query)
    
    resume_scores = {}
    for filename, text in resumes.items():
        text_embedding = embeddings_model.embed_query(text[:2000])
        
        dot_product = sum(a * b for a, b in zip(query_embedding, text_embedding))
        query_norm = sum(a * a for a in query_embedding) ** 0.5
        text_norm = sum(a * a for a in text_embedding) ** 0.5
        
        if query_norm > 0 and text_norm > 0:
            cosine_similarity = dot_product / (query_norm * text_norm)
            resume_scores[filename] = cosine_similarity
    
    if not resume_scores:
        for i, (filename, text) in enumerate(list(resumes.items())[:top_n]):
            resume_scores[filename] = 0.5
    
    sorted_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)
    top_resumes = sorted_resumes[:top_n]
    
    matches = []
    for filename, similarity_score in top_resumes:
        text = resumes[filename]
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
        "top_n": top_n,
        "matches": matches
    }
