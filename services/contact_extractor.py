import re

def extract_contact(text):
    emails = list(set(re.findall(r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b", text)))
    
    phone_patterns = [
        r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        r"\+?\d{10,15}",
        r"\+?\d[\d\s-]{7,13}\d"
    ]
    
    phones = []
    for pattern in phone_patterns:
        found = re.findall(pattern, text)
        phones.extend(found)
    
    phones = list(set(phones))
    
    filtered_phones = []
    for phone in phones:
        phone_stripped = phone.strip()
        
        if re.match(r'^\d{4}\s*[-–]\s*\d{4}$', phone_stripped):
            continue
        
        phone_clean = re.sub(r'[\s\-\(\)\.]', '', phone_stripped)
        
        if len(phone_clean) < 10 or len(phone_clean) > 15:
            continue
        
        if re.match(r'^(19|20)\d{2}', phone_clean[:4]):
            if len(phone_clean) <= 8:
                continue
        
        if re.search(r'[a-zA-Z]', phone_clean):
            continue
        
        filtered_phones.append(phone_stripped)
    
    return {"emails": emails, "phones": filtered_phones}

def extract_name(text):
    lines = text.split('\n')
    
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if not line:
            continue
        
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+[A-Z]\.?)?$',
            r'^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+$',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$'
        ]
        
        for pattern in name_patterns:
            if re.match(pattern, line):
                if not any(keyword in line.lower() for keyword in ['email', 'phone', 'address', 'resume', 'cv', 'objective', 'summary', 'experience', 'education']):
                    if '@' not in line and not re.search(r'\d{4}', line):
                        return line
    
    words = lines[0].strip().split()
    if len(words) >= 2 and len(words) <= 4:
        potential_name = ' '.join(words)
        if all(word[0].isupper() for word in words if word):
            if not any(char.isdigit() for char in potential_name):
                if '@' not in potential_name:
                    return potential_name
    
    return None
