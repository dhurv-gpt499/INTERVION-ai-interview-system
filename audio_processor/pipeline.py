import queue
import torch
import audio_processor.tts_engine as tts_engine

from audio_processor.model_loader import load_models
from audio_processor.mic_capture import AudioCapture
from audio_processor.vad import SpeechSegmenter
from audio_processor.transcriber import transcribe
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState
from llm_interviewer.interviewer import QwenInterviewer
from vision_processor.camera import VideoCaptureThread


def run_pipeline(
    resume_parsed       : dict,
    preferred_companies : list = ["Google", "Microsoft"],
    preferred_roles     : list = ["Software Engineer"],
    target_level        : str  = "entry",
    domain              : str  = "software engineering",
    duration_minutes    : int  = 20,
    # UI callbacks — optional, called to update the interface
    on_state_change     = None,   # fn(state_name: str)
    on_transcript       = None,   # fn(text: str)
    on_qa_complete      = None,   # fn(question: str, answer: str)
    on_vision_scores    = None,   # fn(anxiety: float, confidence: float)
    on_vision_frame     = None,   # fn(frame: np.ndarray)
    is_running          = None,   # fn() -> bool
):
    def notify_state(state_name):
        print(f"[STATE] {state_name}")
        if on_state_change:
            on_state_change(state_name)

    current_anxiety = [0.0]
    def handle_vision_scores(a, c):
        current_anxiety[0] = a
        if on_vision_scores:
            on_vision_scores(a, c)

    whisper, silero_vad = load_models()

    speech_queue = queue.Queue()
    sm           = InterviewStateMachine()
    interviewer  = QwenInterviewer()

    # ----------------------------------------------------------------
    # Start interview — TTS speaks opening question
    # ----------------------------------------------------------------
    print("\n[INTERVIEW STARTING]\n")
    notify_state("Loading AI Model (takes ~10s)...")
    sm.transition(InterviewState.AI_SPEAKING)

    tts_engine.stream_from_llm(
        interviewer.start(
            resume_parsed       = resume_parsed,
            preferred_companies = preferred_companies,
            preferred_roles     = preferred_roles,
            target_level        = target_level,
            domain              = domain,
            duration_minutes    = duration_minutes,
        ),
        on_sentence = lambda s: on_transcript(f"🤖: {s}") if on_transcript else None
    )

    sm.transition(InterviewState.QUESTION_ASKED)
    notify_state("question_asked")
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
                if on_transcript and text:
                    on_transcript(text)

            # finalize — copies complete partial → final
            sm.finalize_answer()
            sm.transition(InterviewState.ANSWER_COMPLETE)
            notify_state("answer_complete")

            print(f"\n[FINAL ANSWER] {sm.final_answer}")
            print(f"[DURATION]     {sm.answer_duration:.1f}s")
            print(f"[PAUSES]       {sm.pause_count}")

            # send answer to Qwen → stream response → TTS speaks
            sm.transition(InterviewState.EVALUATING)
            notify_state("evaluating")
            
            final_text = sm.final_answer
            if current_anxiety[0] > 60:
                final_text += "\n[SYSTEM: Candidate anxiety is high. Soften your tone and be encouraging.]"

            response_generator = interviewer.receive_answer(final_text)

            sm.transition(InterviewState.AI_SPEAKING)
            notify_state("ai_speaking")
            tts_engine.stream_from_llm(
                response_generator,
                on_sentence = lambda s: on_transcript(f"🤖: {s}") if on_transcript else None
            )

            # update current question in state machine
            if interviewer.messages and interviewer.messages[-1]["role"] == "assistant":
                sm.set_question(interviewer.messages[-1]["content"])

            # check if interview concluded
            if not interviewer.is_active:
                sm.transition(InterviewState.SESSION_COMPLETE)
                notify_state("session_complete")
                print("\n[SESSION COMPLETE]")
                return

            # save Q&A, notify UI, reset for next answer
            sm.save_to_history()
            if on_qa_complete and sm.qa_history:
                last = sm.qa_history[-1]
                on_qa_complete(last["question"], last["answer"])

            segmenter.reset()
            sm.transition(InterviewState.QUESTION_ASKED)
            notify_state("question_asked")

        elif event == "force_chunk":
            if not speech_queue.empty():
                audio = speech_queue.get()
                text  = transcribe(audio, whisper)
                sm.append_transcript(text)
                if on_transcript and text:
                    on_transcript(text)
                print(f"\n[PARTIAL] {sm.partial_answer}")

        elif event == "no_answer_silence":
            sm.transition(InterviewState.NO_ANSWER)
            notify_state("no_answer")
            print("\n[NO ANSWER] Comforting candidate...")

            sm.transition(InterviewState.AI_SPEAKING)
            notify_state("ai_speaking")
            tts_engine.stream_from_llm(
                interviewer.receive_answer("...silence..."),
                on_sentence = lambda s: on_transcript(f"🤖: {s}") if on_transcript else None
            )
            sm.transition(InterviewState.QUESTION_ASKED)
            notify_state("question_asked")

    # ----------------------------------------------------------------
    # Setup mic + VAD + Camera
    # ----------------------------------------------------------------
    segmenter = SpeechSegmenter(speech_queue, on_event=on_event)
    mic       = AudioCapture(chunk_size=512)
    
    cam = VideoCaptureThread(
        on_frame=on_vision_frame,
        on_scores=handle_vision_scores
    )
    
    mic.start()
    cam.start()

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    try:
        while True:
            # respect stop signal from UI
            if is_running and not is_running():
                break

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
                notify_state("listening")

            segmenter.process_frame(frame, prob)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        mic.stop()
        cam.stop()
        tts_engine.stop()
        print(f"\nPipeline stopped. Elapsed: {interviewer.elapsed_minutes():.1f} min")
        print("\nFull Q&A History:")
        for i, qa in enumerate(sm.qa_history):
            print(f"\nQ{i+1}: {qa['question']}")
            print(f"A:   {qa['answer']}")
            print(f"     Duration: {qa['duration_sec']}s | Pauses: {qa['pause_count']}")


if __name__ == "__main__":
    mock_resume = {
        "education"   : "B.Tech Computer Science, 2025",
        "skills"      : "Python, PyTorch, Machine Learning, FastAPI",
        "experience"  : "Intern at XYZ Corp — deepfake detection",
        "projects"    : "Deepfake detection using XceptionNet",
        "achievements": "Top 5% Codeforces",
        "competitive" : "Codeforces rating 1450"
    }
    run_pipeline(resume_parsed=mock_resume)
