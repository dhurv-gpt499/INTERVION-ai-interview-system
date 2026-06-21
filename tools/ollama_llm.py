import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini" 

def call_ollama(prompt , system_prompt , temperature = 0.7) :
    full_prompt = f"{system_prompt}\n\n{prompt}"
     
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Ollama. Is it running?"
    except requests.exceptions.Timeout:
        return "ERROR: Ollama took too long to respond."
    except Exception as e:
        return f"ERROR: {e}"
    
def generate_question(context: str, conversation_history: list, difficulty: str = "medium") -> str:

    system_prompt = (
        "You are a senior technical interviewer. Given the candidate's resume context, "
        "ask ONE relevant, conversational interview question. Do not repeat previous questions. "
        f"Difficulty level: {difficulty}."
    )

    history_text = "\n".join(conversation_history) if conversation_history else "No questions asked yet."

    prompt = (
        f"Resume context:\n{context}\n\n"
        f"Previously asked questions:\n{history_text}\n\n"
        f"Now ask the next interview question."
    )

    return call_ollama(prompt, system_prompt=system_prompt)

    
if __name__ == "__main__":
    sample_context = "Candidate has 2 years experience in Python, worked on a deepfake detection project using PyTorch and XceptionNet."
    history = []

    question = generate_question(sample_context, history)
    
    print("Generated Question:\n", question)
