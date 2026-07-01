import requests
import json
import threading

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:latest"

class AnswerEvaluator:
    def __init__(self, db_conn=None):
        self.db_conn = db_conn

    def evaluate_async(self, question: str, answer: str, session_id: str = None, callback=None):
        """Runs the evaluation in a background thread so it doesn't block the pipeline."""
        def _run():
            result = self.evaluate_sync(question, answer)
            if callback:
                callback(result)
            
            # TODO: Save result to Database using session_id
            print(f"\n[EVALUATOR] Scores -> Tech: {result.get('technical_score')}/10 | Comm: {result.get('communication_score')}/10")
            print(f"[EVALUATOR] Status -> {result.get('status')}")
            print(f"[EVALUATOR] Feedback -> {result.get('feedback')}\n")
            
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        
    def evaluate_sync(self, question: str, answer: str) -> dict:
        prompt = f"""You are an expert technical interviewer evaluator.
Evaluate the candidate's answer to the following question.

Question: {question}
Candidate's Answer: {answer}

You must respond with ONLY a valid JSON object matching this exact schema:
{{
    "technical_score": <int 1-10>,
    "communication_score": <int 1-10>,
    "feedback": "<short constructive feedback>",
    "status": "<COMPLETE or MIDWAY>"
}}
If the candidate's answer is very short, missing key details, or they seem stuck, set status to "MIDWAY". Otherwise set to "COMPLETE"."""

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2 # low temp for consistent JSON
            }
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "{}")
            return json.loads(content)
        except Exception as e:
            print(f"[EVALUATOR ERROR] {e}")
            return {
                "technical_score": 5,
                "communication_score": 5,
                "feedback": "Error evaluating answer.",
                "status": "COMPLETE"
            }
