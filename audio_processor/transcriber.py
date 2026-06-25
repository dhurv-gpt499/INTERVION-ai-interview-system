import numpy as np

HINGLISH_PROMPT = (
    "This is a technical interview. "
    "Candidate speaks in Hindi and English mixed. "
    "Python, machine learning, projects, experience, kaam, maine, mera."
)

def transcribe(audio, model):
    # audio must be float32 numpy array at 16000Hz
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    segments, info = model.transcribe(
        audio,
        language=None,           # auto detect hi/en
        beam_size=5,
        vad_filter=True,
        initial_prompt=HINGLISH_PROMPT,
        condition_on_previous_text=False,
    )

    # join all segment texts
    text = " ".join(segment.text.strip() for segment in segments).strip()

    print(f"[Transcribed] lang={info.language} conf={info.language_probability:.2f}")
    print(f"[Text] {text}")

    return text