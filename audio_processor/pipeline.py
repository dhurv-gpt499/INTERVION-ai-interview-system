import queue
import torch
from audio_processor.model_loader import load_models
from audio_processor.mic_capture import AudioCapture
from audio_processor.vad import SpeechSegmenter
from audio_processor.transcriber import transcribe
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState


def run_pipeline():
    whisper, silero_vad = load_models()

    speech_queue = queue.Queue()
    sm = InterviewStateMachine()          # ← replaces the old dict

    # ----------------------------------------------------------------
    # VAD event handler
    # ----------------------------------------------------------------
    def on_event(event):

        if event == "post_speech_silence":
            sm.increment_pause()
            sm.finalize_answer()
            sm.transition(InterviewState.ANSWER_COMPLETE)

            # pull audio from queue and transcribe final chunk
            if not speech_queue.empty():
                audio = speech_queue.get()
                text  = transcribe(audio, whisper)
                sm.append_transcript(text)

            sm.save_to_history()
            print(f"\n[FINAL ANSWER] {sm.final_answer}")
            print(f"[DURATION]     {sm.answer_duration:.1f}s")
            print(f"[PAUSES]       {sm.pause_count}")
            print(sm.status())

            # reset segmenter for next question
            segmenter.reset()

        elif event == "force_chunk":
            # mid-answer chunk — transcribe and append, don't finalize
            if not speech_queue.empty():
                audio = speech_queue.get()
                text  = transcribe(audio, whisper)
                sm.append_transcript(text)
                print(f"\n[PARTIAL] {sm.partial_answer}")

        elif event == "no_answer_silence":
            sm.transition(InterviewState.NO_ANSWER)
            print("\n[STATE] Candidate not responding — would comfort here")

    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    segmenter = SpeechSegmenter(speech_queue, on_event=on_event)
    mic       = AudioCapture(chunk_size=512)

    # start session
    first_question = "Tell me about yourself"
    sm.set_question(first_question)
    sm.transition(InterviewState.QUESTION_ASKED)

    mic.start()
    print(f"\nQuestion: {first_question}")
    print("Listening...\n")

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    try:
        while True:
            frame = mic.get_chunk()

            if len(frame) != 512:
                continue

            tensor = torch.from_numpy(frame).float()
            with torch.no_grad():
                prob = silero_vad(tensor, 16000).item()

            # detect speech START → transition to LISTENING
            is_speech = prob > 0.5
            if is_speech and sm.current_state == InterviewState.QUESTION_ASKED:
                sm.start_answer()
                sm.transition(InterviewState.LISTENING)

            segmenter.process_frame(frame, prob)

    except KeyboardInterrupt:
        mic.stop()
        print("\nPipeline stopped.")
        print("\nFull Q&A History:")
        for i, qa in enumerate(sm.qa_history):
            print(f"\nQ{i+1}: {qa['question']}")
            print(f"A:   {qa['answer']}")
            print(f"     Duration: {qa['duration_sec']}s | Pauses: {qa['pause_count']}")


if __name__ == "__main__":
    run_pipeline()