import re

def extract_contact(text):
    emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b", text)
    phones = re.findall(r"\+?\d[\d\s-]{8,14}\d", text)
    return {"emails": emails, "phones": phones}
