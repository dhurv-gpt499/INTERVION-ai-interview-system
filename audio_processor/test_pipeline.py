import queue
import torch
import time
from model_loader import load_models
from mic_capture import AudioCapture
from vad import SpeechSegmenter
from transcriber import transcribe

print("Loading models...")
whisper, silero_vad = load_models()
print("Models ready.\n")

speech_queue = queue.Queue()
frame_count = [0]

def on_event(event):
    print(f"\n[EVENT] >>> {event.upper()} <<<")

    if event in ("post_speech_silence", "force_chunk"):
        if not speech_queue.empty():
            audio = speech_queue.get()
            duration = len(audio) / 16000
            print(f"[CHUNK] Duration: {duration:.2f}s | Samples: {len(audio)}")
            text = transcribe(audio, whisper)
            print(f"[TEXT]  {text}")
        else:
            print("[CHUNK] Queue empty — nothing to transcribe")

    elif event == "no_answer_silence":
        print("[INFO]  Candidate not responding — would comfort here")

    print()

segmenter = SpeechSegmenter(speech_queue, on_event=on_event)
mic = AudioCapture(chunk_size=512)
mic.start()

print("Speak into your mic. Ctrl+C to stop.\n")
print("-" * 50)

try:
    while True:
        frame = mic.get_chunk()

        if len(frame) != 512:
            continue

        frame_count[0] += 1

        tensor = torch.from_numpy(frame).float()
        with torch.no_grad():
            prob = silero_vad(tensor, 16000).item()

        # print every 10 frames so you can see VAD working
        if frame_count[0] % 10 == 0:
            bar = "#" * int(prob * 20)
            print(f"\r[VAD] {prob:.2f} |{bar:<20}|", end="", flush=True)

        segmenter.process_frame(frame, prob)

except KeyboardInterrupt:
    mic.stop()
    print("\n\nTest stopped.")