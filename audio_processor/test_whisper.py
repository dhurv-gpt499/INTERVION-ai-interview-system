from faster_whisper import WhisperModel


model = WhisperModel("small", device="cuda", compute_type="int8_float16")
 
 
segments, info = model.transcribe("audio_processor/sample.mp3" , language="en" , beam_size=5, vad_filter=True)


print("Detected language:", info.language)
print("Confidence:", info.language_probability)
print()


for segment in segments:
    print(f"[{segment.start:.1f}s -> {segment.end:.1f}s]  {segment.text}")