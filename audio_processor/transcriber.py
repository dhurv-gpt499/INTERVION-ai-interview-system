import numpy as np

HINGLISH_PROMPT = (
    "This is a technical interview. Candidate speaks in English. "
    "Technical terms: XceptionNet, PyTorch, quantization, FastAPI, "
    "deepfake detection, CNN, machine learning, Python, neural network,Web development, REST API, backend, frontend, database, SQL, NoSQL "
    "model optimization, inference, dataset, accuracy, preprocessing."
)

def transcribe(audio, model):
    segments, info = model.transcribe(
        audio,
        language              = "en",       # ← force English, remove None
        beam_size             = 5,
        vad_filter            = True,
        initial_prompt        = HINGLISH_PROMPT,
        condition_on_previous_text = False,
    )

    text = " ".join(segment.text.strip() for segment in segments).strip()

    # filter hallucinations — discard if mostly non-ASCII
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    if ascii_ratio < 0.8:
        print("[Transcribed] Hallucination detected — discarded")
        return ""

    # filter very low confidence
    if info.language_probability < 0.4:
        print(f"[Transcribed] Low confidence ({info.language_probability:.2f}) — discarded")
        return ""

    print(f"[Transcribed] lang={info.language} conf={info.language_probability:.2f}")
    print(f"[Text] {text}")
    return text