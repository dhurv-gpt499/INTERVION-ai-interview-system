import queue
import torch
from audio_processor.model_loader import load_models
from audio_processor.mic_capture import AudioCapture
from audio_processor.vad import SpeechSegmenter
from audio_processor.transcriber import transcribe
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState
from llm_interviewer import interviewer



import queue
import torch
import audio_processor.tts_engine as tts_engine

from audio_processor.model_loader import load_models
from audio_processor.mic_capture import AudioCapture
from audio_processor.vad import SpeechSegmenter
from audio_processor.transcriber import transcribe
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState
from llm_interviewer.interviewer import QwenInterviewer


def run_pipeline(resume_parsed: dict):

    whisper, silero_vad = load_models()

    speech_queue = queue.Queue()
    sm           = InterviewStateMachine()
    interviewer  = QwenInterviewer()

    # ----------------------------------------------------------------
    # Start interview — TTS speaks opening question
    # ----------------------------------------------------------------
    print("\n[INTERVIEW STARTING]\n")
    sm.transition(InterviewState.AI_SPEAKING)

    tts_engine.stream_from_llm(
        interviewer.start(
            resume_parsed       = resume_parsed,
            preferred_companies = ["Google", "Microsoft"],
            preferred_roles     = ["ML Engineer"],
            target_level        = "entry",
            domain              = "software engineering",
            duration_minutes    = 20,
        )
    )

    sm.transition(InterviewState.QUESTION_ASKED)
    print("Listening...\n")

    # ----------------------------------------------------------------
    # VAD event handler
    # ----------------------------------------------------------------
    def on_event(event):

        if event == "post_speech_silence":
            sm.increment_pause()

            # transcribe FIRST — before finalizing
            if not speech_queue.empty():
                audio = speech_queue.get()
                text  = transcribe(audio, whisper)
                sm.append_transcript(text)

            # finalize — copies complete partial → final
            sm.finalize_answer()
            sm.transition(InterviewState.ANSWER_COMPLETE)

            print(f"\n[FINAL ANSWER] {sm.final_answer}")
            print(f"[DURATION]     {sm.answer_duration:.1f}s")
            print(f"[PAUSES]       {sm.pause_count}")

            # send answer to Qwen → stream response → TTS speaks
            sm.transition(InterviewState.EVALUATING)
            response_generator = interviewer.receive_answer(sm.final_answer)

            sm.transition(InterviewState.AI_SPEAKING)
            tts_engine.stream_from_llm(response_generator)
            
            # update current question in state machine
            if interviewer.messages and interviewer.messages[-1]["role"] == "assistant":
                    sm.set_question(interviewer.messages[-1]["content"])
            # check if interview concluded
            if not interviewer.is_active:
                sm.transition(InterviewState.SESSION_COMPLETE)
                print("\n[SESSION COMPLETE]")
                return

            # save Q&A and reset for next answer
            sm.save_to_history()
            segmenter.reset()
            sm.transition(InterviewState.QUESTION_ASKED)

        elif event == "force_chunk":
            # mid-answer chunk — transcribe and append only
            if not speech_queue.empty():
                audio = speech_queue.get()
                text  = transcribe(audio, whisper)
                sm.append_transcript(text)
                print(f"\n[PARTIAL] {sm.partial_answer}")

        elif event == "no_answer_silence":
            sm.transition(InterviewState.NO_ANSWER)
            print("\n[NO ANSWER] Comforting candidate...")

            sm.transition(InterviewState.AI_SPEAKING)
            tts_engine.stream_from_llm(
                interviewer.receive_answer("...silence...")
            )
            sm.transition(InterviewState.QUESTION_ASKED)

    # ----------------------------------------------------------------
    # Setup mic + VAD
    # ----------------------------------------------------------------
    segmenter = SpeechSegmenter(speech_queue, on_event=on_event)
    mic       = AudioCapture(chunk_size=512)
    mic.start()

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
        print(f"Elapsed: {interviewer.elapsed_minutes():.1f} min")
        print("\nFull Q&A History:")
        for i, qa in enumerate(sm.qa_history):
            print(f"\nQ{i+1}: {qa['question']}")
            print(f"A:   {qa['answer']}")
            print(f"     Duration: {qa['duration_sec']}s | Pauses: {qa['pause_count']}")


if __name__ == "__main__":
    # replace with actual resume parser output
    mock_resume = {
        "education"   : "B.Tech Computer Science, 2025",
        "skills"      : "Python, PyTorch, Machine Learning, FastAPI",
        "experience"  : "Intern at XYZ Corp — deepfake detection",
        "projects"    : "Deepfake detection using XceptionNet",
        "achievements": "Top 5% Codeforces",
        "competitive" : "Codeforces rating 1450"
    }

    run_pipeline(resume_parsed=mock_resume)
