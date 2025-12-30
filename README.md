# ResumeGenie

**Tagline:** “AI-powered resume matching & contact extraction system”

---

## Project Overview

ResumeGenie is a **backend AI system** designed to help recruitment teams automatically **match job requirements with candidate resumes**. The system goes beyond simple keyword search by using **AI embeddings and semantic search** to identify the best-fit candidates. In addition, it extracts **contact information, skills, and experience** from resumes, making it easier to shortlist candidates quickly.

This project is **backend-only**, designed for **portfolio, interview demos, or recruitment automation**, and can be extended in the future with **RAG, multi-agent workflows, and advanced NLP pipelines**.

---

## Key Features

- **AI-based Resume Matching:** Uses embeddings to find the most relevant resumes for any job requirement.  
- **Contact Information Extraction:** Automatically extracts emails and phone numbers from resumes.  
- **Skills Identification:** Detects relevant skills listed in resumes.  
- **Experience Extraction:** Identifies years of experience mentioned in resumes.  
- **RAG-Ready Architecture:** Supports future integration with retrieval-augmented generation for more advanced queries.  
- **Backend-Only System:** Implemented in FastAPI with AI pipelines for easy integration and testing.  

---

## Technology Stack

- **Backend:** Python, FastAPI  
- **AI & Embeddings:** LangChain, OpenAI embeddings  
- **Vector Search:** FAISS  
- **Resume Parsing:** PyPDF2, docx2txt  
- **Information Extraction:** Regex + keyword-based parsing  
- **Optional Extensions:** LangGraph (workflow), CrewAI (multi-agent processing), RAG for contextual retrieval  

---

## How It Works

1. **Job Requirement Input:** The system receives a job description from the user.  
2. **Resume Processing:** Resumes (PDF/DOCX) are converted into text.  
3. **Embedding & Matching:** Each resume is converted into an embedding and matched semantically against the job requirement using vector similarity.  
4. **Information Extraction:** From the top matching resumes, the system extracts emails, phone numbers, skills, and experience.  
5. **Output:** Returns a structured response in JSON format containing the top candidate resumes with extracted information.  

---

## Future Improvements

- Integrate **LangGraph** for workflow orchestration and decision-making.  
- Multi-agent resume analysis using **CrewAI** for parallel skill and experience extraction.  
- Enhance **skill extraction** using NLP entity recognition.  
- Use **RAG + LLM scoring** for more intelligent resume ranking and candidate recommendation.
