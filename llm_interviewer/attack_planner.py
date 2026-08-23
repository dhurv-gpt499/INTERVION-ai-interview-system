import json
import asyncio
import os
import requests

async def generate_attack_plan(resume_text: str, target_company: str, target_role: str, llm_backend: str = "Local (Ollama)", api_key: str = "") -> dict:
    """
    Analyzes the candidate's resume against the target company and role, identifying weaknesses,
    exaggerations, and generating a hyper-specific interrogation strategy.
    Runs asynchronously during the 30-second websocket loading phase.
    """
    print(f"[ATTACK PLANNER] Formulating attack plan for {target_company} ({target_role})...")
    
    system_prompt = """You are a Principal Engineering Interview Architect for top-tier tech firms (FAANG/Quant). 
Your job is to read a candidate's resume and generate a ruthless, highly-targeted "Attack Plan" for the upcoming interview.
You must find the weaknesses, the exaggerated claims, and the technical gaps in their resume, and prescribe exactly how to grill them.

Output ONLY a raw JSON object with the following schema, and no other text or markdown formatting:
{
  "candidate_weaknesses": ["weakness 1", "weakness 2"],
  "exaggerated_claims_to_probe": ["claim 1 to probe deep into", "claim 2"],
  "interrogation_strategy": "A 2-sentence description of the aggressive tone and strategy to use (e.g. 'Drill relentlessly into their distributed systems claim by asking how they handled network partitions.')",
  "recommended_topics": ["topic1", "topic2", "topic3"]
}"""

    user_prompt = f"TARGET COMPANY: {target_company}\nTARGET ROLE: {target_role}\n\nRESUME:\n{resume_text}"

    fallback_needed = False
    result = {}

    if llm_backend == "Cloud API (Groq)" and api_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            # Run blocking request in thread pool
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: requests.post(url, json=payload, headers=headers, timeout=20))
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"]
            result = json.loads(text)
        except Exception as e:
            print(f"[ATTACK PLANNER] Groq failed: {e}. Falling back to Ollama.")
            fallback_needed = True
    else:
        fallback_needed = True

    if fallback_needed:
        OLLAMA_URL = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen2.5:latest",
            "prompt": system_prompt + "\n\n" + user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 8192
            }
        }
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: requests.post(OLLAMA_URL, json=payload, timeout=30))
            res.raise_for_status()
            text = res.json().get("response", "")
            
            # Robust JSON extraction handling Qwen's <think> tags
            import re
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                result = json.loads(text)
        except Exception as e:
            print(f"[ATTACK PLANNER] Extraction error: {e}")
            result = {
                "candidate_weaknesses": ["General algorithmic depth"],
                "exaggerated_claims_to_probe": ["Impact of major projects"],
                "interrogation_strategy": "Maintain a rigorous, standard technical bar. Probe the candidate on why they chose their tech stack.",
                "recommended_topics": ["System Design", "Data Structures"]
            }

    print(f"[ATTACK PLANNER] Strategy ready. Targets: {result.get('recommended_topics', [])}")
    return result
