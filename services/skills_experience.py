from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

def extract_skills_experience(resume_text, llm):
    prompt_template = ChatPromptTemplate.from_template(
        """Extract skills and work experience from the following resume text. 
        Return ONLY a valid JSON object with the following structure:
        {{
            "skills": ["skill1", "skill2", "skill3"],
            "experience_years": number or null,
            "work_experience": [
                {{
                    "company": "company name",
                    "position": "job title",
                    "duration": "time period",
                    "description": "brief description"
                }}
            ]
        }}
        
        Resume Text:
        {resume_text}
        
        Return only the JSON object, no other text:"""
    )
    
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"resume_text": resume_text[:3000]})
        
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return {
                "skills": data.get("skills", []),
                "experience_years": data.get("experience_years"),
                "work_experience": data.get("work_experience", [])
            }
        else:
            return {
                "skills": [],
                "experience_years": None,
                "work_experience": []
            }
    except Exception as e:
        return {
            "skills": [],
            "experience_years": None,
            "work_experience": []
        }

