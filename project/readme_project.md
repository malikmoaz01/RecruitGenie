# RecruitGenie - Complete Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [File Structure & Responsibilities](#file-structure--responsibilities)
4. [Technology Stack](#technology-stack)
5. [Complete Workflow](#complete-workflow)
6. [API Endpoints](#api-endpoints)
7. [Configuration](#configuration)
8. [Key Components Explained](#key-components-explained)

---

## 🎯 Project Overview

**RecruitGenie** is an AI-powered resume matching and contact extraction system built with FastAPI. It uses semantic search, embeddings, and RAG (Retrieval-Augmented Generation) to match job descriptions with candidate resumes. The system automatically extracts contact information, skills, and work experience from resumes.

### Main Purpose
- **Resume Matching**: Find the best candidates for a job using AI-powered semantic similarity
- **Information Extraction**: Automatically extract emails, phone numbers, names, skills, and experience
- **RAG-based Querying**: Answer questions about resumes using chunked document retrieval
- **Parallel Processing**: Use LangGraph workflows for efficient multi-agent resume analysis

---

## 🏗️ Project Architecture

```
RecruitGenie/
├── main.py                          # FastAPI application & main entry point
├── config.py                        # Configuration & environment variables
├── requirements.txt                 # Python dependencies
├── services/                        # Core service modules
│   ├── parser.py                    # Resume parsing (PDF/DOCX to text)
│   ├── contact_extractor.py         # Email, phone, name extraction
│   ├── embeddings.py                # Vector store creation & retrieval
│   ├── chunking.py                 # Resume text chunking for RAG
│   └── skills_experience.py        # Skills & experience extraction using LLM
├── langgraph_workflows/             # Multi-agent workflow orchestration
│   └── resume_analysis_workflow.py  # LangGraph workflow for parallel processing
└── data/resumes/                    # Input folder for resume files (PDF/DOCX)
```

---

## 📁 File Structure & Responsibilities

### 1. **main.py** - Main Application Entry Point

**Purpose**: FastAPI application that orchestrates the entire system.

**Key Responsibilities**:
- Initialize FastAPI server
- Load and parse all resumes from `data/resumes/` folder
- Create vector store from chunked resume documents
- Set up RAG (Retrieval-Augmented Generation) chain for querying
- Handle `/match-resumes` POST endpoint
- Calculate similarity scores between job descriptions and resumes
- Process top matching resumes using LangGraph workflow or sequential processing

**Key Variables**:
- `CHUNK_SIZE = 3000`: Size of text chunks (~750 tokens)
- `CHUNK_OVERLAP = 400`: Overlap between chunks
- `TOP_K_CHUNKS = 5`: Number of chunks to retrieve for RAG
- `SIMILARITY_THRESHOLD = 0.5`: Minimum similarity score (0.0 to 1.0)
- `USE_LANGGRAPH_WORKFLOW = True`: Toggle between LangGraph and sequential processing

**Main Functions**:
- `root()`: Welcome endpoint
- `match_resumes()`: Main matching endpoint that processes job descriptions

**Flow in main.py**:
1. Load environment variables and API keys
2. Extract all resumes from `data/resumes/` folder
3. Chunk resumes into smaller pieces for better retrieval
4. Create FAISS vector store from chunks
5. Create retriever with similarity threshold
6. Build RAG chain (retriever → prompt → LLM → output parser)
7. When `/match-resumes` is called:
   - Get RAG result for the query
   - Calculate similarity scores for all resumes
   - Filter by threshold and get top N matches
   - Process each match using LangGraph workflow or sequential extraction
   - Return structured JSON response

---

### 2. **config.py** - Configuration Management

**Purpose**: Centralized configuration for API keys and service URLs.

**Key Variables**:
- `API_KEY`: OpenAI/OpenRouter API key from environment
- `OPENROUTER_BASE_URL`: Base URL for OpenRouter API
- `OPENROUTER_API_KEY`: API key for OpenRouter

**Usage**: Loaded by `main.py` and other modules to configure LLM and embeddings.

---

### 3. **services/parser.py** - Resume Parsing Service

**Purpose**: Extract text from PDF and DOCX resume files.

**Key Functions**:
- `extract_text(file_path)`: Extracts text from a single file (PDF or DOCX)
  - Uses `PyPDF2` for PDF files
  - Uses `docx2txt` for DOCX files
- `extract_all_resumes(resume_folder)`: Processes all resumes in a folder
  - Returns dictionary: `{filename: text_content}`

**Technologies Used**:
- `PyPDF2`: PDF text extraction
- `docx2txt`: DOCX text extraction

**Flow**:
1. Scan `data/resumes/` folder
2. For each PDF/DOCX file, extract text
3. Return dictionary mapping filenames to text content

---

### 4. **services/contact_extractor.py** - Contact Information Extraction

**Purpose**: Extract emails, phone numbers, and names from resume text.

**Key Functions**:
- `extract_contact(text)`: Extracts emails and phone numbers
  - **Emails**: Uses regex pattern `r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b"`
  - **Phones**: Uses multiple regex patterns to handle different formats
  - Filters out invalid phone numbers (dates, too short/long, contains letters)
  - Returns: `{"emails": [...], "phones": [...]}`

- `extract_name(text)`: Extracts candidate name from resume
  - Looks at first 10 lines of resume
  - Uses pattern matching for common name formats
  - Filters out lines containing keywords like "email", "phone", "resume", etc.
  - Returns name string or None

**Technologies Used**:
- Python `re` (regex) module for pattern matching

**Flow**:
1. Scan resume text for email patterns
2. Scan for phone number patterns (multiple formats)
3. Filter and validate phone numbers
4. Extract name from top of resume using pattern matching

---

### 5. **services/chunking.py** - Resume Chunking Service

**Purpose**: Split large resumes into smaller chunks for better RAG retrieval.

**Why Chunking?**
- Large resumes don't fit in LLM context windows efficiently
- Chunking allows retrieving only relevant parts of resumes
- Better similarity matching at granular level

**Key Functions**:
- `_split_text_recursive(text, chunk_size, chunk_overlap, separators)`: 
  - Recursively splits text by separators (`\n\n`, `\n`, `. `, ` `)
  - Tries to break at natural boundaries (paragraphs, sentences)
  - Applies overlap between chunks for context preservation

- `chunk_resumes(resumes, chunk_size=3000, chunk_overlap=400)`:
  - Processes all resumes in dictionary
  - Creates chunks with metadata (filename, chunk_index, total_chunks)
  - Returns list of dictionaries: `[{"text": "...", "metadata": {...}}]`

- `chunk_single_resume(text, filename, ...)`: Chunks a single resume

**Default Parameters**:
- `chunk_size = 3000`: ~750 tokens per chunk
- `chunk_overlap = 400`: Overlap to maintain context

**Flow**:
1. For each resume, split text into chunks
2. Try to break at natural boundaries (paragraphs, sentences)
3. Add metadata to each chunk (source file, index, etc.)
4. Return list of chunked documents

---

### 6. **services/embeddings.py** - Vector Store & Embeddings

**Purpose**: Create vector embeddings and FAISS vector store for semantic search.

**Key Functions**:
- `create_vector_store(resume_texts=None, chunked_documents=None)`:
  - Creates FAISS vector store from resume texts or chunked documents
  - Uses `HuggingFaceEmbeddings` with model `sentence-transformers/all-MiniLM-L6-v2`
  - Converts text to vector embeddings (384-dimensional vectors)
  - Returns FAISS vector store object

- `create_tuned_retriever(vector_store, top_k=5, search_type="similarity", similarity_threshold=None)`:
  - Creates a retriever from vector store
  - Configures top_k (number of chunks to retrieve)
  - Sets similarity threshold for filtering
  - Returns configured retriever

**Technologies Used**:
- `langchain_huggingface.HuggingFaceEmbeddings`: For creating embeddings
- `langchain_community.vectorstores.FAISS`: For vector storage and similarity search
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)

**Flow**:
1. Load embeddings model (HuggingFace)
2. Convert chunked documents to Document objects
3. Create FAISS vector store from documents
4. Create retriever with configured parameters (top_k, threshold)

---

### 7. **services/skills_experience.py** - Skills & Experience Extraction

**Purpose**: Extract skills, experience years, and work history using LLM.

**Key Functions**:
- `extract_skills_experience(resume_text, llm)`:
  - Uses LLM (ChatOpenAI) to extract structured information
  - Prompts LLM to return JSON with:
    - `skills`: List of skills
    - `experience_years`: Total years of experience
    - `work_experience`: List of work history entries (company, position, duration, description)
  - Parses JSON response using regex
  - Returns structured dictionary

**Technologies Used**:
- `langchain_core.prompts.ChatPromptTemplate`: For prompt templating
- `langchain_core.output_parsers.StrOutputParser`: For parsing LLM output
- LLM (ChatOpenAI) for intelligent extraction

**Flow**:
1. Create prompt template requesting JSON output
2. Invoke LLM with resume text (first 3000 chars)
3. Extract JSON from LLM response using regex
4. Parse JSON and return structured data

---

### 8. **langgraph_workflows/resume_analysis_workflow.py** - Multi-Agent Workflow

**Purpose**: Orchestrate parallel processing of resume analysis using LangGraph.

**Why LangGraph?**
- Modular workflow design
- Easy to parallelize tasks
- State management between nodes
- Can be extended for complex decision-making

**Key Components**:

**State Class**:
- `ResumeAnalysisState`: TypedDict that holds state between workflow nodes
  - Contains: resume_text, filename, job_description, contact_info, skills_experience, similarity_score, embeddings_model, llm, final_result

**Workflow Nodes**:
1. **`contact_extraction_node`**: Extracts contact info (name, emails, phones)
2. **`skills_experience_node`**: Extracts skills and work experience using LLM
3. **`similarity_calculation_node`**: Calculates cosine similarity with job description
4. **`merge_results_node`**: Combines all extracted information into final result

**Workflow Graph**:
```
Entry → extract_contact → extract_skills_experience → calculate_similarity → merge_results → END
```

**Key Functions**:
- `create_resume_analysis_workflow(llm)`: Creates and compiles LangGraph workflow
- `analyze_resume_parallel(...)`: Analyzes single resume using workflow
- `analyze_multiple_resumes_parallel(...)`: Analyzes multiple resumes

**Technologies Used**:
- `langgraph.graph.StateGraph`: For workflow orchestration
- `langchain_openai.ChatOpenAI`: For LLM calls

**Flow**:
1. Initialize workflow with state
2. Run contact extraction node
3. Run skills/experience extraction node
4. Run similarity calculation node
5. Merge all results in final node
6. Return structured result

---

## 🔧 Technology Stack

### Core Framework
- **FastAPI**: Web framework for building REST API
- **Python 3.12**: Programming language

### AI & ML
- **LangChain**: Framework for LLM applications
- **LangGraph**: Workflow orchestration for multi-agent systems
- **OpenAI/OpenRouter**: LLM API for text generation
- **HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- **FAISS**: Vector database for similarity search

### Document Processing
- **PyPDF2**: PDF text extraction
- **docx2txt**: DOCX text extraction

### Utilities
- **python-dotenv**: Environment variable management
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server for FastAPI

---

## 🔄 Complete Workflow

### Initialization Phase (When Server Starts)

1. **Load Configuration** (`main.py` lines 18-21)
   - Load `.env` file
   - Set `OPENAI_API_KEY` from config

2. **Extract Resumes** (`main.py` line 28)
   - Call `extract_all_resumes("data/resumes")`
   - `parser.py` scans folder and extracts text from PDF/DOCX files
   - Returns dictionary: `{filename: text}`

3. **Chunk Resumes** (`main.py` lines 51-55)
   - Call `chunk_resumes(resumes, chunk_size=3000, chunk_overlap=400)`
   - `chunking.py` splits each resume into smaller chunks
   - Returns list of chunked documents with metadata

4. **Create Vector Store** (`main.py` line 58)
   - Call `create_vector_store(chunked_documents=chunked_documents)`
   - `embeddings.py` creates FAISS vector store from chunks
   - Uses HuggingFace embeddings model

5. **Create Retriever** (`main.py` lines 72-77)
   - Call `create_tuned_retriever(vector_store, top_k=5, similarity_threshold=0.5)`
   - Configures retriever for RAG queries

6. **Build RAG Chain** (`main.py` lines 79-109)
   - Creates prompt template
   - Builds chain: `retriever → format_docs → prompt → llm → output_parser`
   - `format_docs` groups chunks by source file

### Request Processing Phase (When `/match-resumes` is Called)

1. **Receive Request** (`main.py` line 137)
   - POST request with `JobRequest` containing:
     - `job_description`: Job requirements text
     - `top_n`: Number of top matches (default: 5)
     - Optional overrides: `use_langgraph`, `chunk_size`, `top_k_chunks`, `similarity_threshold`

2. **Get RAG Result** (`main.py` line 149)
   - Invoke RAG chain with job description
   - Retrieves relevant chunks and generates answer

3. **Calculate Similarity Scores** (`main.py` lines 152-191)
   - Create embedding for job description query
   - Retrieve relevant chunks using retriever
   - For each chunk:
     - Calculate cosine similarity with query
     - Track maximum similarity per resume
   - For resumes not in retrieved chunks:
     - Calculate similarity from full text (first 2000 chars)

4. **Filter & Sort** (`main.py` lines 194-205)
   - Filter resumes by similarity threshold (default: 0.5)
   - Sort by similarity score (descending)
   - Get top N resumes

5. **Process Top Resumes** (`main.py` lines 208-239)
   
   **Option A: LangGraph Workflow** (`USE_LANGGRAPH_WORKFLOW = True`)
   - For each top resume:
     - Call `analyze_resume_parallel(resume_text, filename, job_description, embeddings_model, llm)`
     - Workflow runs:
       1. Contact extraction node → extracts name, emails, phones
       2. Skills/experience node → extracts skills, experience years, work history
       3. Similarity calculation node → calculates cosine similarity
       4. Merge results node → combines all data
     - Returns structured result with similarity score

   **Option B: Sequential Processing** (`USE_LANGGRAPH_WORKFLOW = False`)
   - For each top resume:
     - Call `extract_contact(text)` → get emails, phones
     - Call `extract_name(text)` → get name
     - Call `extract_skills_experience(text, llm)` → get skills, experience
     - Combine results manually

6. **Return Response** (`main.py` lines 241-253)
   - JSON response containing:
     - `rag_result`: Answer from RAG chain
     - `total_resumes`: Total number of resumes processed
     - `total_chunks`: Total number of chunks created
     - `top_n`: Number of matches returned
     - `rag_config`: Configuration used
     - `matches`: Array of top matching resumes with:
       - name, similarity_score, emails, phones, skills, experience_years, work_experience

---

## 🌐 API Endpoints

### 1. **GET /** - Root Endpoint
- **Purpose**: Welcome message and API information
- **Response**: 
  ```json
  {
    "message": "Welcome to ResumeGenie API",
    "endpoints": {...},
    "status": "running"
  }
  ```

### 2. **POST /match-resumes** - Main Matching Endpoint

**Request Body**:
```json
{
  "job_description": "Looking for a Python developer with 3+ years experience...",
  "top_n": 5,
  "use_langgraph": true,  // Optional: override default
  "chunk_size": 3000,     // Optional: override default
  "top_k_chunks": 5,      // Optional: override default
  "similarity_threshold": 0.5  // Optional: override default
}
```

**Response**:
```json
{
  "rag_result": "Based on the resumes...",
  "total_resumes": 4,
  "total_chunks": 12,
  "top_n": 5,
  "rag_config": {
    "chunk_size": 3000,
    "top_k_chunks": 5,
    "similarity_threshold": 0.5,
    "use_langgraph_workflow": true
  },
  "matches": [
    {
      "name": "John Doe",
      "similarity_score": 0.8234,
      "emails": ["john@example.com"],
      "phones": ["+1234567890"],
      "skills": ["Python", "FastAPI", "ML"],
      "experience_years": 5,
      "work_experience": [...]
    }
  ]
}
```

---

## ⚙️ Configuration

### Environment Variables (`.env` file)
```
API_KEY=your_openai_or_openrouter_api_key
```

### Configurable Parameters in `main.py`
- `CHUNK_SIZE = 3000`: Size of text chunks (~750 tokens)
- `CHUNK_OVERLAP = 400`: Overlap between chunks
- `TOP_K_CHUNKS = 5`: Number of chunks to retrieve for RAG
- `SIMILARITY_THRESHOLD = 0.5`: Minimum similarity score (0.0 to 1.0)
- `USE_LANGGRAPH_WORKFLOW = True`: Use LangGraph workflow or sequential processing

### Embeddings Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Device: CPU (configurable in `embeddings.py`)

---

## 🔑 Key Components Explained

### RAG (Retrieval-Augmented Generation)
- **Purpose**: Answer questions about resumes using retrieved context
- **How it works**:
  1. Job description query is converted to embedding
  2. Similar chunks are retrieved from vector store
  3. Retrieved chunks are passed to LLM as context
  4. LLM generates answer based on context

### Vector Similarity Search
- **Purpose**: Find resumes semantically similar to job description
- **How it works**:
  1. Convert job description to embedding vector
  2. Convert resume chunks to embedding vectors
  3. Calculate cosine similarity between vectors
  4. Rank resumes by similarity score

### Chunking Strategy
- **Why**: Large resumes don't fit in LLM context efficiently
- **How**: Split resumes into 3000-character chunks with 400-character overlap
- **Benefits**: 
  - Better retrieval of relevant sections
  - More granular similarity matching
  - Efficient use of LLM context window

### LangGraph Workflow
- **Purpose**: Modular, parallelizable resume analysis
- **Nodes**: Contact extraction, skills extraction, similarity calculation, result merging
- **Benefits**: 
  - Easy to extend with new agents
  - Can be parallelized for multiple resumes
  - Clear separation of concerns

---

## 📊 Data Flow Diagram

```
User Request (Job Description)
    ↓
[main.py] Receive POST /match-resumes
    ↓
[main.py] Get RAG result from chunked documents
    ↓
[embeddings.py] Calculate similarity scores
    ↓
[main.py] Filter & sort top N resumes
    ↓
[langgraph_workflows] OR [services] Process each resume
    ├─→ [contact_extractor.py] Extract contact info
    ├─→ [skills_experience.py] Extract skills & experience
    └─→ [embeddings.py] Calculate similarity
    ↓
[main.py] Return JSON response with matches
```

---

## 🚀 How to Use

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Create `.env` file:
   ```
   API_KEY=your_api_key_here
   ```

3. **Add Resumes**:
   Place PDF/DOCX files in `data/resumes/` folder

4. **Run Server**:
   ```bash
   uvicorn main:app --reload
   ```

5. **Test API**:
   - Visit `http://localhost:8000/docs` for Swagger UI
   - Or send POST request to `/match-resumes` with job description

---

## 📝 Notes

- The system processes all resumes at startup and creates vector store
- Chunking improves retrieval accuracy for large resumes
- LangGraph workflow can be toggled on/off via `USE_LANGGRAPH_WORKFLOW`
- Similarity threshold filters out low-quality matches
- RAG provides contextual answers about resumes
- All extraction uses regex (contact) or LLM (skills/experience)

---

## 🔮 Future Enhancements

- Multi-agent CrewAI integration for advanced analysis
- Enhanced skill extraction using NLP entity recognition
- RAG + LLM scoring for intelligent ranking
- Support for more file formats (TXT, RTF)
- Database integration for storing resume data
- User authentication and job posting management
- Real-time resume processing (not just at startup)

---

**Last Updated**: 2025
**Project**: RecruitGenie - AI-Powered Resume Matching System

