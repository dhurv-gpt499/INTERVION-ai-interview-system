import time
import json
import requests
from .question_generator import decide_categories

OLLAMA_URL = "http://localhost:11434/api/chat"   # ← chat endpoint
MODEL_NAME = "qwen2.5:latest"


class QwenInterviewer:
    def __init__(self, llm_backend="Local (Ollama)", api_key=""):
        self.messages      = []     # full conversation history
        self.start_time    = None
        self.duration_sec  = 0
        self.turn_count    = 0
        self.is_active     = False
        self.rag           = None
        self.llm_backend   = llm_backend
        self.api_key       = api_key

    def start(
        self,
        resume_parsed: dict,
        preferred_companies: list,
        preferred_roles: list,
        target_level: str,
        domain: str,
        duration_minutes: int = 20,
        past_weak_areas: list = [],
        past_covered_topics: list = [],
    ):
        from .build_system_prompt import build_interviewer_system_prompt

        # Generate custom categories based on resume parsing and target company profile
        ordered_topics = decide_categories(resume_parsed, target_level, preferred_companies)

        system_prompt = build_interviewer_system_prompt(
            resume_parsed        = resume_parsed,
            preferred_companies  = preferred_companies,
            preferred_roles      = preferred_roles,
            target_level         = target_level,
            domain               = domain,
            duration_minutes     = duration_minutes,
            ordered_topics       = ordered_topics,
            past_weak_areas      = past_weak_areas,
            past_covered_topics  = past_covered_topics,
        )

        # load system prompt as first message
        self.messages     = [{"role": "system", "content": system_prompt}]
        self.start_time   = time.time()
        self.duration_sec = duration_minutes * 60
        self.turn_count   = 0
        self.is_active    = True
        self.target_company = preferred_companies[0] if preferred_companies else ""

        # Initialize and build RAG index
        from .rag_engine import ResumeRAG
        self.rag = ResumeRAG()
        if resume_parsed:
            self.rag.build_index(resume_parsed)

        # get opening question (streaming)
        return self._stream_response()


    def receive_answer(self, answer_text: str):
        if not self.is_active:
            return iter([])

        # check time before responding
        if self.is_time_up():
            return self.send_timesup()

        clean_text = answer_text.strip()
        ephemeral_hint = ""

        # Check if candidate is silent or didn't answer
        if clean_text in ["", "no_answer", "no_answer_silence", "silence", "[SILENCE - CANDIDATE PAUSED]"]:
            self.messages.append({"role": "user", "content": "[SILENCE - CANDIDATE PAUSED]"})
            ephemeral_hint = (
                "[CANDIDATE SILENCE / HESITATION DETECTED]: The candidate has paused and seems stuck on your previous question. "
                "Do NOT change the topic or ask a new question. Using the evaluation rubric, give the candidate a brief, "
                "supportive technical hint or simplify your previous question to help them progress."
            )
        else:
            self.messages.append({"role": "user", "content": answer_text})
            # Fetch RAG context based on what the user just said
            rag_context = ""
            if self.rag and getattr(self.rag, "resume_vectors", None) is not None:
                last_q = self.messages[-2]["content"] if len(self.messages) >= 2 else ""
                query = f"{last_q} {answer_text}"
                rag_context = self.rag.get_relevant_context(query, target_company=getattr(self, 'target_company', ''))
            if rag_context:
                ephemeral_hint = f"[SYSTEM INJECTION - KNOWLEDGE GRAPH & RUBRICS]: Use the following verified details and grading standard to ground your evaluation and formulate your next probing question:\n{rag_context}"

        self.turn_count += 1
        return self._stream_response(ephemeral_hint=ephemeral_hint)


    def send_timesup(self):
        self.messages.append({
            "role": "user",
            "content": "TIME'S UP"
        })
        return self._stream_response()


    def _stream_response(self, ephemeral_hint: str = ""):
        messages_payload = list(self.messages)
        if ephemeral_hint:
            messages_payload.append({"role": "system", "content": ephemeral_hint})

        full_response = ""

        # Use Cloud API directly if selected
        if self.llm_backend == "Cloud API (Groq)":
            import os
            api_key = self.api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                yield "I'm sorry, you selected Cloud API but no Groq API Key was provided."
                return
            
            try:
                fb_response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={"model": "llama-3.1-8b-instant", "messages": messages_payload, "stream": True, "temperature": 0.7},
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    stream=True, timeout=10
                )
                fb_response.raise_for_status()
                for line in fb_response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]": break
                            try:
                                token = json.loads(data_str)["choices"][0].get("delta", {}).get("content", "")

                                if token:
                                    full_response += token
                                    yield token
                            except Exception: pass
            except Exception as e:
                yield f"Cloud API Error: {e}"
                return
            
            self.messages.append({"role": "assistant", "content": full_response})
            if "That concludes our interview" in full_response:
                self.is_active = False
            return

        payload = {
            "model"   : MODEL_NAME,
            "messages": messages_payload,
            "stream"  : True,
            "options" : {
                "temperature": 0.7,
                "num_ctx"    : 4096,
            }
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json    = payload,
                stream  = True,
                timeout = 180
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                full_response += token
                yield token

                if chunk.get("done", False):
                    break

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError):
            import os
            api_key = self.api_key or os.environ.get("GROQ_API_KEY")
            if not api_key:
                yield "I'm sorry, I cannot connect to the local Ollama instance, and no GROQ_API_KEY environment variable was found for the fallback API. Please start Ollama or set your GROQ_API_KEY."
                return

            fallback_url = "https://api.groq.com/openai/v1/chat/completions"
            fallback_payload = {
                "model": "llama-3.1-8b-instant",
                "messages": messages_payload,
                "stream": True,
                "temperature": 0.7
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            try:
                fb_response = requests.post(fallback_url, json=fallback_payload, headers=headers, stream=True, timeout=10)
                fb_response.raise_for_status()
                for line in fb_response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            data_str = decoded[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                token = chunk["choices"][0].get("delta", {}).get("content", "")
                                if token:
                                    full_response += token
                                    yield token
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                yield f"Fallback API Error: {e}"
                return
        except Exception as e:
            yield f"Error: {e}"
            return

        # save full response to history
        self.messages.append({"role": "assistant", "content": full_response})

        # check if interview concluded
        if "That concludes our interview" in full_response:
            self.is_active = False


    def is_time_up(self) -> bool:
        if not self.start_time:
            return False
        return (time.time() - self.start_time) >= self.duration_sec

    def elapsed_minutes(self) -> float:
        if not self.start_time:
            return 0.0
        return (time.time() - self.start_time) / 60

    def get_history(self) -> list:
        """Returns Q&A pairs for database storage without ephemeral injections."""
        qa = []
        msgs = self.messages[1:]   # skip system prompt
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "assistant" and msgs[i+1]["role"] == "user":
                qa.append({
                    "question": msgs[i]["content"],
                    "answer"  : msgs[i+1]["content"],
                })
        return qa