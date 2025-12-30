from PyPDF2 import PdfReader
import docx2txt
import os

def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
    return text

def extract_all_resumes(resume_folder="data/resumes"):
    resumes = {}
    if not os.path.exists(resume_folder):
        return resumes
    for file_name in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file_name)
        if os.path.isfile(file_path) and (file_path.endswith(".pdf") or file_path.endswith(".docx")):
            try:
                text = extract_text(file_path)
                if text.strip():
                    resumes[file_name] = text
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue
    return resumes
