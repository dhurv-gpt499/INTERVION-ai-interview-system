import fitz 
import spacy
import re
import json
from pathlib import Path

nlp = spacy.load("en_core_web_sm") #using the model

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path) # Extracting Text from PDF
    full_text = ""
    
    for page in doc:
        full_text += page.get_text()
    
    doc.close()
    return full_text

def extract_personal_info(text: str) -> dict:
    """Extract contact info from raw resume text — name comes from user input."""
    
    lines = text.split('\n')
    
    email = ""
    phone = ""
    github = ""
    linkedin = ""
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Email — has # prefix or @ in line
        if line_stripped.startswith('#'):
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line_stripped)
            if email_match:
                email = email_match.group()
        elif '@' in line_stripped:
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line_stripped)
            if email_match:
                email = email_match.group()
        
        # Phone — has phone unicode prefix or +91
        if line_stripped.startswith('+91') or line_stripped.startswith('\uf095'):
            phone_match = re.search(r'\+91[\-\s]?\d{10}|\+91[\-\s]?\d{5}[\-\s]?\d{5}', line_stripped)
            if phone_match:
                phone = phone_match.group()
        # fallback — line contains +91
        if not phone and '+91' in line_stripped:
            phone_match = re.search(r'\+91[\-\s]?\d{10}', line_stripped)
            if phone_match:
                phone = phone_match.group()
        
        # GitHub — has § prefix
        if line_stripped.startswith('§'):
            # Get the handle after § symbol
            handle = line_stripped.replace('§', '').strip()
            # Make sure it looks like a github handle not a bullet point
            if handle and len(handle) > 3 and ' ' not in handle:
                github = handle
        
        # LinkedIn — has ï prefix
        if line_stripped.startswith('ï'):
            handle = line_stripped.replace('ï', '').strip()
            if handle and len(handle) > 3 and ' ' not in handle:
                linkedin = handle
    
    return {
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github
    }

def extract_education(text: str) -> list:
    """Extract all education entries from resume."""
    
    education = []
    lines = text.split('\n')
    
    # Find education section boundaries
    edu_start = -1
    edu_end = -1
    end_headers = ['experience', 'skills', 'projects', 'achievements', 
                   'certifications', 'technical']
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if 'education' in line_lower and len(line_lower) < 20:
            edu_start = i
        if edu_start != -1 and i > edu_start + 2:
            if any(header in line_lower for header in end_headers):
                edu_end = i
                break
    
    if edu_start == -1:
        return education
        
    edu_end = edu_end if edu_end != -1 else edu_start + 25
    edu_lines = lines[edu_start:edu_end]
    edu_text = '\n'.join(edu_lines)
    
    # Patterns
    cgpa_pattern = r'CGPA[:\s]*(\d+\.?\d*)'
    percentage_pattern = r'(\d{2,3}(?:\.\d{1,2})?)\s*%'
    year_pattern = r'\b(20\d{2})\b'
    degree_pattern = r'\b(Bachelor of Technology|B\.?Tech|Master of Technology|M\.?Tech|B\.?E|BCA|MCA|B\.?Sc|M\.?Sc|MBA|Ph\.?D)\b'
    branch_pattern = r'\b(Computer Science Engineering|Computer Science|CSE|Information Technology|IT|Electronics|ECE|Mechanical|Civil|Data Science|AI|Machine Learning)\b'
    
    # Split into blocks by empty lines — each block is one education entry
    blocks = []
    current_block = []
    
    for line in edu_lines[1:]:  # skip the "Education" header line
        if line.strip() == '':
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        else:
            current_block.append(line.strip())
    if current_block:
        blocks.append('\n'.join(current_block))
    
    for block in blocks:
        if not block.strip():
            continue
            
        doc = nlp(block)
        
        # Extract org names via spaCy
        organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        
        # If spaCy missed it, take first line of block as institution
        first_line = block.split('\n')[0].strip()
        institution = organizations[0] if organizations else first_line
        
        cgpa = re.findall(cgpa_pattern, block, re.IGNORECASE)
        percentage = re.findall(percentage_pattern, block)
        years = re.findall(year_pattern, block)
        degrees = re.findall(degree_pattern, block, re.IGNORECASE)
        branches = re.findall(branch_pattern, block, re.IGNORECASE)
        
        edu_entry = {
            "institution": institution,
            "degree": degrees[0] if degrees else "",
            "branch": branches[0] if branches else "",
            "cgpa": cgpa[0] if cgpa else "",
            "percentage": percentage[0] if percentage else "",
            "years": years,
        }
        
        education.append(edu_entry)
    
    return education