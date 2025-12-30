from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from services.parser import extract_all_resumes
from services.contact_extractor import extract_contact
from services.embeddings import create_vector_store
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
if not resumes:
    resume_texts = []
    vector_store = None
    qa = None
else:
    resume_texts = list(resumes.values())
    resume_names = list(resumes.keys())
    vector_store = create_vector_store(resume_texts)
    
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

@app.get("/match-resumes", response_class=HTMLResponse)
def match_resumes_get():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ResumeGenie - Match Resumes</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            form {
                margin-top: 20px;
            }
            textarea {
                width: 100%;
                min-height: 150px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #0056b3;
            }
            .info {
                background-color: #e7f3ff;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 20px;
                border-left: 4px solid #007bff;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 ResumeGenie - Match Resumes</h1>
            <div class="info">
                <strong>Note:</strong> Enter a job description below to find matching resumes. 
                You can also use the API directly via POST request or check the <a href="/docs">Swagger UI</a>.
            </div>
            <form id="matchForm">
                <label for="job_description"><strong>Job Description:</strong></label>
                <textarea id="job_description" name="job_description" placeholder="Enter job description here...&#10;Example: Looking for Python developer with FastAPI experience"></textarea>
                <br>
                <button type="submit">Match Resumes</button>
            </form>
            <div id="result" class="result" style="display:none;"></div>
        </div>
        <script>
            document.getElementById('matchForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const jobDescription = document.getElementById('job_description').value;
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.textContent = 'Loading...';
                
                try {
                    const response = await fetch('/match-resumes', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ job_description: jobDescription })
                    });
                    
                    const data = await response.json();
                    resultDiv.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    resultDiv.textContent = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/match-resumes")
def match_resumes(request: JobRequest):
    query = request.job_description
    
    if not resumes:
        return {"error": "No resumes found in data/resumes folder", "matches": []}
    
    if qa is None:
        return {"error": "RAG chain not initialized", "matches": []}
    
    result_text = qa.invoke(query)
    matches = []
    for name, text in resumes.items():
        contact = extract_contact(text)
        matches.append({
            "name": name,
            "emails": contact["emails"],
            "phones": contact["phones"]
        })
    return {"rag_result": result_text, "matches": matches}
