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
    """Extract just the skills to support RAG, and pass the raw resume directly to avoid JSON hell."""
    
    prompt = f"""List the top 10 technical skills, programming languages, and frameworks mentioned in this resume as a comma-separated list. 
DO NOT OUTPUT JSON. ONLY OUTPUT A COMMA-SEPARATED LIST OF SKILLS. NO OTHER TEXT.

Resume:
{markdown_text}"""

    text = ""
    if llm_backend == "Cloud API (Groq)":
        if not api_key:
            print("Groq selected but no API key provided. Falling back to Ollama.")
        else:
            try:
                import os
                import requests
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "openai/gpt-oss-120b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                }
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                text = res.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Groq API Error: {e}. Falling back to Ollama.")

    if not text:
        # Fallback / Local (Ollama)
        import requests
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
            res = requests.post(OLLAMA_URL, json=payload, timeout=60)
            res.raise_for_status()
            text = res.json().get("response", "")
        except Exception as e:
            print(f"Extraction error: {e}")
            text = ""
            
    # Clean text
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    skills = [s.strip() for s in text.split(",") if s.strip()]
    
    return {
        "raw_resume": markdown_text,
        "skills": {"Extracted": skills},
        "competitive": "leetcode" in markdown_text.lower() or "codeforces" in markdown_text.lower()
    }


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