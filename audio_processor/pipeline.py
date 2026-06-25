import queue
import torch
from model_loader import load_models
from mic_capture import AudioCapture
from vad import SpeechSegmenter

def run_pipeline():
    whisper, silero_vad = load_models()
    speech_queue = queue.Queue()
    state = {
        "question"        : "Tell me about yourself",
        "partial_answer"  : "",
        "final_answer"    : "",
        "answer_duration" : 0.0,
        "pause_count"     : 0,
    }

    def on_event(event):
        pass

    segmenter = SpeechSegmenter(speech_queue, on_event=on_event)
    
    mic = AudioCapture(chunk_size=512) 
    mic.start()

    print(f"Question: {state['question']}")
    print("Listening...\n")

    try:
        while True:
            frame = mic.get_chunk()
            
            if len(frame) != 512:
                continue 

            tensor = torch.from_numpy(frame).float().unsqueeze(0)
            
            with torch.no_grad():
                prob = silero_vad(tensor, 16000).item()

            segmenter.process_frame(frame, prob)

    except KeyboardInterrupt:
        mic.stop()
        print("Pipeline stopped.")

if __name__ == "__main__":
    run_pipeline()