import json
import re
import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
import html 

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"

def extract_text(pdf_path: str) -> str:
    """Extract clean markdown from PDF using Docling."""
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def llm_extract(markdown_text: str, llm_backend: str = "Local (Ollama)", api_key: str = "") -> dict:
    """Send markdown to LLM to extract structured JSON."""
    
    prompt = f"""You are a resume parser. Extract information from the resume markdown below and return ONLY a valid JSON object. No explanation, no markdown, no code blocks. Just raw JSON.

Important rules:
- Return ONLY the JSON object, nothing before or after it
- For skills languages: include ALL programming languages e.g. Python, Java, C++, SQL
- For skills frameworks: platforms and frameworks e.g. PyTorch, TensorFlow, React, Django
- For skills tools: tools only e.g. Git, Docker, VS Code, Linux
- For skills databases: only databases e.g. Oracle, PostgreSQL, MongoDB
- For skills cloud: only cloud platforms e.g. AWS, Azure, GCP
- CGPA is out of 10 e.g. 8.3. Percentage is out of 100 e.g. 74.0. Never mix them. If value above 20 it is percentage
- Extract ALL education entries including school and college
- Extract ALL work experience entries even if responsibilities are empty
- For experience year_start extract only start date. For year_end extract only end date or "Present"
- For project descriptions include every bullet point as separate array item
- Extract ALL certifications into achievements array as plain strings
- Extract competitive programming profiles — leetcode, codeforces, hackerrank, codechef usernames
- For candidate_archetype: classify candidate into exactly ONE primary domain e.g. "Quant & Low-Latency Systems", "Full-Stack Web Development", "AI & Machine Learning", "Data Architecture & ETL", "Cybersecurity", or "General Software Engineering"
- For flagship_project_tech: extract an array of 5 to 10 core technical keywords from the candidate's #1 most impressive project
- For probe_targets: identify 2 or 3 vague claims, bold architectural statements, or missing evidence from the resume that a strict FAANG/Quant interviewer should challenge
- If field is missing use empty string or empty array

Return exactly this structure:
{{
  "personal_info": {{
    "email": "",
    "phone": "",
    "github": "",
    "linkedin": ""
  }},
  "candidate_archetype": "",
  "flagship_project_tech": [],
  "probe_targets": [],
  "education": [
    {{
      "institution": "",
      "degree": "",
      "branch": "",
      "cgpa": "",
      "percentage": "",
      "year_start": "",
      "year_end": ""
    }}
  ],
  "skills": {{
    "languages": [],
    "frameworks": [],
    "tools": [],
    "databases": [],
    "cloud": []
  }},
  "experience": [
    {{
      "company": "",
      "role": "",
      "year_start": "",
      "year_end": "",
      "responsibilities": []
    }}
  ],
  "projects": [
    {{
      "name": "",
      "tech_stack": [],
      "description": [],
      "github_link": ""
    }}
  ],
  "competitive_programming": {{
    "leetcode": "",
    "codeforces": "",
    "hackerrank": "",
    "codechef": ""
  }},
  "achievements": []
}}

Resume:
{markdown_text}"""

    if llm_backend == "Cloud API (Groq)":
        if not api_key:
            print("Groq selected but no API key provided. Falling back to Ollama.")
        else:
            try:
                import os
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "system", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1
                }
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                return json.loads(res.json()["choices"][0]["message"]["content"], strict=False)
            except Exception as e:
                print(f"Groq API Error: {e}. Falling back to Ollama.")

    # Fallback / Local (Ollama)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        
        raw_response = response.json()["response"].strip()
        
        # Clean markdown fences
        raw_response = re.sub(r'^```json\s*', '', raw_response)
        raw_response = re.sub(r'^```\s*', '', raw_response)
        raw_response = re.sub(r'\s*```$', '', raw_response)
        raw_response = raw_response.strip()
        
        # Extract only JSON object
        start = raw_response.find('{')
        end = raw_response.rfind('}')
        if start != -1 and end != -1:
            raw_response = raw_response[start:end+1]
        
        return json.loads(raw_response, strict=False)

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"LLM returned: {raw_response[:300]}")
        return retry_extract(markdown_text, llm_backend, api_key)
    
    except requests.exceptions.ConnectionError:
        print("Ollama not running! Run: ollama serve")
        return {}
    
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}


def retry_extract(markdown_text: str, llm_backend: str = "Local (Ollama)", api_key: str = "") -> dict:
    """Fallback with stricter prompt at temperature 0."""
    
    prompt = f"""Extract resume data as JSON only. Return nothing except the JSON object.

{{
  "personal_info": {{"email": "", "phone": "", "github": "", "linkedin": ""}},
  "candidate_archetype": "",
  "flagship_project_tech": [],
  "probe_targets": [],
  "education": [{{"institution": "", "degree": "", "branch": "", "cgpa": "", "percentage": "", "year_start": "", "year_end": ""}}],
  "skills": {{"languages": [], "frameworks": [], "tools": [], "databases": [], "cloud": []}},
  "experience": [{{"company": "", "role": "", "year_start": "", "year_end": "", "responsibilities": []}}],
  "projects": [{{"name": "", "tech_stack": [], "description": [], "github_link": ""}}],
  "competitive_programming": {{"leetcode": "", "codeforces": "", "hackerrank": "", "codechef": ""}},
  "achievements": []
}}

Resume:
{markdown_text}"""

    if llm_backend == "Cloud API (Groq)":
        if api_key:
            try:
                import os
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "qwen/qwen3.6-27b",
                    "messages": [{"role": "system", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0
                }
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                return json.loads(res.json()["choices"][0]["message"]["content"], strict=False)
            except Exception as e:
                print(f"Groq API Error: {e}. Falling back to Ollama.")

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192}
        }, timeout=180)
        
        raw = response.json()["response"].strip()
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        return json.loads(raw, strict=False)
    
    except Exception as e:
        print(f"Retry failed: {e}")
        return {}

def post_process(markdown_text: str, parsed: dict) -> dict:
    """Fix small consistent issues LLM gets wrong."""
    import html
    import re
    
    # Fix 1 — GitHub from markdown links
    if not parsed.get("personal_info", {}).get("github"):
        github_match = re.search(r'https?://github\.com/[\w\-]+', markdown_text)
        if github_match:
            if "personal_info" not in parsed:
                parsed["personal_info"] = {}
            parsed["personal_info"]["github"] = github_match.group()
    
    # Fix 2 — Clean HTML entities from all string values recursively
    def clean_html(obj):
        if isinstance(obj, str):
            return html.unescape(obj)
        elif isinstance(obj, list):
            return [clean_html(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: clean_html(v) for k, v in obj.items()}
        return obj
    parsed = clean_html(parsed)
    
    # Fix 3 — Cognizant role merged into company name
    for exp in parsed.get("experience", []):
        company = exp.get("company", "")
        if not exp.get("role") and company:
            # Common job titles that get merged into company
            titles = ["Associate", "Analyst", "Engineer", "Manager", 
                     "Consultant", "Developer", "Intern"]
            for title in titles:
                if company.endswith(title):
                    exp["role"] = title
                    exp["company"] = company[:-len(title)].strip().rstrip(',')
                    break
    
    return parsed


def parse_resume(pdf_path: str, llm_backend: str = "Local (Ollama)", api_key: str = ""):
    print("Extracting text with Docling...")
    markdown_text = extract_text(pdf_path)
    
    if not markdown_text.strip():
        print("No text extracted!")
        return {}, ""
    
    print(f"Sending to LLM ({llm_backend}) for parsing...")
    parsed = llm_extract(markdown_text, llm_backend, api_key)
    
    if not parsed:
        return {}, markdown_text
    
    # Post processing
    parsed = post_process(markdown_text, parsed)
    
    print("Resume parsed successfully!")
    return parsed, markdown_text


if __name__ == "__main__":
    result, markdown = parse_resume("resume_parser/test_resume.pdf")
    print(json.dumps(result, indent=2))