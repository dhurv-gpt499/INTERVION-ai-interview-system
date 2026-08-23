import os
import io
import asyncio
import numpy as np
import soundfile as sf
from typing import Optional

# Run blocking transcription in a thread pool
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

class STTEngine:
    def __init__(self, use_groq=False, api_key=None):
        self.use_groq = use_groq
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.whisper_model = None
        
        if not self.use_groq:
            # We delay loading to avoid blocking on startup if possible
            pass

    def _load_local_model(self):
        if self.whisper_model is None:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = WhisperModel("base", device=device, compute_type="int8" if device == "cpu" else "float16")

    async def transcribe_async(self, audio_data: np.ndarray) -> str:
        """
        Takes 16kHz float32 audio data and returns the transcription.
        """
        if len(audio_data) == 0:
            return ""
            
        loop = asyncio.get_running_loop()
        
        if self.use_groq and self.api_key:
            return await loop.run_in_executor(executor, self._transcribe_groq, audio_data)
        else:
            return await loop.run_in_executor(executor, self._transcribe_local, audio_data)

    def _transcribe_local(self, audio_data: np.ndarray) -> str:
        self._load_local_model()
        segments, _ = self.whisper_model.transcribe(audio_data, beam_size=5, language="en", condition_on_previous_text=False)
        text = "".join(segment.text for segment in segments)
        return text.strip()

    def _transcribe_groq(self, audio_data: np.ndarray) -> str:
        try:
            import requests
            
            # Convert float32 numpy array to wav bytes
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_data, 16000, format="WAV", subtype="PCM_16")
            wav_bytes = wav_io.getvalue()

            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            files = {
                "file": ("audio.wav", wav_bytes, "audio/wav")
            }
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "json",
                "language": "en"
            }
            response = requests.post(url, headers=headers, files=files, data=data, timeout=15)
            response.raise_for_status()
            
            return response.json().get("text", "").strip()
        except Exception as e:
            print(f"[GROQ STT ERROR] {e}")
            return self._transcribe_local(audio_data)

