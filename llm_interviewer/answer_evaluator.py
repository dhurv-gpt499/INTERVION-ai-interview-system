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

    def evaluate_final_interview(
        self,
        qa_history: list,
        resume_parsed: dict,
        preferred_companies: list,
        target_level: str,
        avg_anxiety: float = 0.0,
        avg_confidence: float = 100.0
    ) -> dict:
        """Evaluates the complete interview transcript and returns a structured JSON report card."""
        if not qa_history:
            return {
                "overall_score": 0,
                "verdict": "No Data",
                "summary": "No interview questions or answers were recorded during this session.",
                "strengths": [],
                "weaknesses": [],
                "topic_scores": {
                    "Technical Depth & Architecture": 0,
                    "Problem Solving & Logic": 0,
                    "Communication & Clarity": 0,
                    "Composure & Confidence": 0
                },
                "key_recommendation": "Complete a full interview session to receive evaluation feedback."
            }

        companies_str = ", ".join(preferred_companies) if preferred_companies else "top tech companies"
        
        transcript_lines = []
        for i, qa in enumerate(qa_history):
            transcript_lines.append(f"Turn {i+1}:\nQ: {qa.get('q', '')}\nA: {qa.get('a', '')}\n")
        transcript_str = "\n".join(transcript_lines)

        r = resume_parsed or {}
        resume_summary = f"Skills: {r.get('skills', '')}\nProjects: {r.get('projects', '')}\nExperience: {r.get('experience', '')}"

        prompt = f"""You are a Principal Engineering Hiring Committee evaluating a candidate for {target_level} roles at {companies_str}.
Here is the candidate's background summary:
{resume_summary}

Here is the complete interview Q&A transcript:
{transcript_str}

Real-time vision & body language analytics during the interview showed:
- Average Confidence Score: {avg_confidence:.1f}%
- Average Anxiety Score: {avg_anxiety:.1f}%

Evaluate the candidate's performance across the entire interview. Be objective, rigorous, and direct (no sugarcoating).
You MUST respond with ONLY a valid JSON object matching this exact schema:
{{
    "overall_score": <int 1-100, where 75+ is hire quality>,
    "verdict": "<Strong Hire | Hire | Leaning Hire | Leaning No Hire | No Hire>",
    "summary": "<2-3 sentence executive evaluation summary>",
    "strengths": [
        "<specific technical strength demonstrated>",
        "<another strength demonstrated>"
    ],
    "weaknesses": [
        "<specific area where answers lacked depth or accuracy>",
        "<another area for improvement>"
    ],
    "topic_scores": {{
        "Technical Depth & Architecture": <int 1-100>,
        "Problem Solving & Logic": <int 1-100>,
        "Communication & Clarity": <int 1-100>,
        "Composure & Confidence": <int 1-100>
    }},
    "key_recommendation": "<1 actionable piece of engineering advice for their next interview>"
}}"""

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192
            }
        }

        try:
            print("[EVALUATOR] Generating final interview evaluation report card...")
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "{}")
            report = json.loads(content)
            print(f"[EVALUATOR] Final evaluation complete -> Score: {report.get('overall_score')}/100 | Verdict: {report.get('verdict')}")
            return report
        except Exception as e:
            print(f"[EVALUATOR ERROR] Failed to generate final evaluation: {e}")
            return {
                "overall_score": 70,
                "verdict": "Evaluation Completed (Fallback Mode)",
                "summary": f"The candidate completed {len(qa_history)} interview turns. Detailed LLM scoring encountered a timeout or format error.",
                "strengths": ["Completed multiple technical interview turns.", "Engaged with interviewer questions."],
                "weaknesses": ["Could provide more architectural detail in follow-up answers."],
                "topic_scores": {
                    "Technical Depth & Architecture": 70,
                    "Problem Solving & Logic": 70,
                    "Communication & Clarity": 75,
                    "Composure & Confidence": int(avg_confidence)
                },
                "key_recommendation": "Review the recorded Q&A transcript to practice deeper technical explanations."
            }

