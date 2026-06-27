import time
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"   # ← chat endpoint
MODEL_NAME = "qwen2.5:latest"


class QwenInterviewer:
    def __init__(self):
        self.messages      = []     # full conversation history
        self.start_time    = None
        self.duration_sec  = 0
        self.turn_count    = 0
        self.is_active     = False

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
        from build_system_prompt import build_interviewer_system_prompt

        system_prompt = build_interviewer_system_prompt(
            resume_parsed        = resume_parsed,
            preferred_companies  = preferred_companies,
            preferred_roles      = preferred_roles,
            target_level         = target_level,
            domain               = domain,
            duration_minutes     = duration_minutes,
            past_weak_areas      = past_weak_areas,
            past_covered_topics  = past_covered_topics,
        )

        # load system prompt as first message
        self.messages     = [{"role": "system", "content": system_prompt}]
        self.start_time   = time.time()
        self.duration_sec = duration_minutes * 60
        self.turn_count   = 0
        self.is_active    = True

        # get opening question (streaming)
        return self._stream_response()


    def receive_answer(self, answer_text: str):
        if not self.is_active:
            return

        # add candidate answer to history
        self.messages.append({"role": "user", "content": answer_text})

        # check time before responding
        if self.is_time_up():
            return self.send_timesup()

        self.turn_count += 1
        return self._stream_response()


    def send_timesup(self):
        self.messages.append({
            "role": "user",
            "content": "TIME'S UP"
        })
        return self._stream_response()


    def _stream_response(self):
        payload = {
            "model"   : MODEL_NAME,
            "messages": self.messages,
            "stream"  : True,
            "options" : {
                "temperature": 0.7,
                "num_ctx"    : 4096,
            }
        }

        full_response = ""

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
                yield token              # ← feed directly to tts_engine

                if chunk.get("done", False):
                    break

        except requests.exceptions.ConnectionError:
            yield "I'm sorry, there seems to be a technical issue. Please wait."
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
        """Returns Q&A pairs for database storage."""
        qa = []
        msgs = self.messages[1:]   # skip system prompt
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "assistant" and msgs[i+1]["role"] == "user":
                qa.append({
                    "question": msgs[i]["content"],
                    "answer"  : msgs[i+1]["content"],
                })
        return qa