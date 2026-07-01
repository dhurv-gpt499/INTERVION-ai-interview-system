import queue
import threading
import torch
import audio_processor.tts_engine as tts_engine

from audio_processor.model_loader import load_models
from audio_processor.mic_capture import AudioCapture
from audio_processor.vad import SpeechSegmenter
from audio_processor.transcriber import transcribe
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState
from llm_interviewer.interviewer import QwenInterviewer
from llm_interviewer.answer_evaluator import AnswerEvaluator
from vision_processor.camera import VideoCaptureThread

def run_pipeline(
    resume_parsed       : dict,
    preferred_companies : list = ["Google", "Microsoft"],
    preferred_roles     : list = ["Software Engineer"],
    target_level        : str  = "entry",
    domain              : str  = "software engineering",
    duration_minutes    : int  = 20,
    # UI callbacks
    on_state_change     = None,
    on_transcript       = None,
    on_qa_complete      = None,
    on_vision_scores    = None,
    on_vision_frame     = None,
    is_running          = None,
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

    speech_queue  = queue.Queue()
    command_queue = queue.Queue()
    llm_queue     = queue.Queue()
    
    sm            = InterviewStateMachine()
    interviewer   = QwenInterviewer()
    evaluator     = AnswerEvaluator()

    # ----------------------------------------------------------------
    # Background Worker Threads
    # ----------------------------------------------------------------
    def stt_worker():
        """Handles transcription of audio chunks so mic loop doesn't block."""
        while True:
            cmd = command_queue.get()
            if cmd == "stop":
                break
                
            if cmd == "post_speech_silence":
                if not speech_queue.empty():
                    audio = speech_queue.get()
                    text  = transcribe(audio, whisper)
                    sm.append_transcript(text)
                    if on_transcript and text:
                        on_transcript(text)

                sm.finalize_answer()
                sm.transition(InterviewState.ANSWER_COMPLETE)
                notify_state("answer_complete")
                
                print(f"\n[FINAL ANSWER] {sm.final_answer}")
                llm_queue.put(("evaluate", sm.final_answer))

            elif cmd == "force_chunk":
                if not speech_queue.empty():
                    audio = speech_queue.get()
                    text  = transcribe(audio, whisper)
                    sm.append_transcript(text)
                    if on_transcript and text:
                        on_transcript(text)
                    print(f"\n[PARTIAL] {sm.partial_answer}")

    def llm_worker():
        """Handles LLM generation and evaluation without blocking UI/Mic."""
        while True:
            cmd, payload = llm_queue.get()
            if cmd == "stop":
                break
                
            if cmd == "evaluate":
                sm.transition(InterviewState.EVALUATING)
                notify_state("evaluating")
                
                # 1. Fire off background evaluator
                evaluator.evaluate_async(sm.current_question, payload)
                
                # 2. Add dynamic context to payload
                final_text = payload
                if current_anxiety[0] > 60:
                    final_text += "\n[SYSTEM: Candidate anxiety is high. Soften your tone and be encouraging.]"

                # 3. Stream Qwen Response
                response_generator = interviewer.receive_answer(final_text)

                sm.transition(InterviewState.AI_SPEAKING)
                notify_state("ai_speaking")
                tts_engine.stream_from_llm(
                    response_generator,
                    on_sentence = lambda s: on_transcript(f"🤖: {s}") if on_transcript else None
                )

                # 4. Check interview state
                if not interviewer.is_active:
                    sm.transition(InterviewState.SESSION_COMPLETE)
                    notify_state("session_complete")
                    print("\n[SESSION COMPLETE]")
                    continue

                if interviewer.messages and interviewer.messages[-1]["role"] == "assistant":
                    sm.set_question(interviewer.messages[-1]["content"])

                sm.save_to_history()
                if on_qa_complete and sm.qa_history:
                    last = sm.qa_history[-1]
                    on_qa_complete(last["question"], last["answer"])

                # 5. Reset segmenter so it waits for new input safely
                segmenter.reset()
                sm.transition(InterviewState.QUESTION_ASKED)
                notify_state("question_asked")
                
            elif cmd == "no_answer":
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

    # Start Daemons
    t_stt = threading.Thread(target=stt_worker, daemon=True)
    t_llm = threading.Thread(target=llm_worker, daemon=True)
    t_stt.start()
    t_llm.start()

    # ----------------------------------------------------------------
    # Start interview — TTS speaks opening question
    # ----------------------------------------------------------------
    print("\n[INTERVIEW STARTING]\n")
    notify_state("Loading AI Model (takes ~10s)...")
    sm.transition(InterviewState.AI_SPEAKING)

    # Initial boot text
    init_gen = interviewer.start(
        resume_parsed       = resume_parsed,
        preferred_companies = preferred_companies,
        preferred_roles     = preferred_roles,
        target_level        = target_level,
        domain              = domain,
        duration_minutes    = duration_minutes,
    )
    tts_engine.stream_from_llm(
        init_gen,
        on_sentence = lambda s: on_transcript(f"🤖: {s}") if on_transcript else None
    )

    if interviewer.messages:
        sm.set_question(interviewer.messages[-1]["content"])

    sm.transition(InterviewState.QUESTION_ASKED)
    notify_state("question_asked")
    print("Listening...\n")

    # ----------------------------------------------------------------
    # Fast VAD event handler (0 Latency Queue Pushes)
    # ----------------------------------------------------------------
    def on_event(event):
        if event == "post_speech_silence":
            sm.increment_pause()
            command_queue.put("post_speech_silence")
        elif event == "force_chunk":
            command_queue.put("force_chunk")
        elif event == "no_answer_silence":
            llm_queue.put(("no_answer", ""))

    segmenter = SpeechSegmenter(speech_queue, on_event=on_event)
    mic       = AudioCapture(chunk_size=512)
    cam       = VideoCaptureThread(
        on_frame  = on_vision_frame,
        on_scores = handle_vision_scores
    )
    
    mic.start()
    cam.start()

    # ----------------------------------------------------------------
    # Main loop (Zero-blocking pure Mic I/O)
    # ----------------------------------------------------------------
    try:
        while True:
            # UI Stop Signal
            if is_running and not is_running():
                break

            frame = mic.get_chunk()
            if len(frame) != 512:
                continue

            tensor = torch.from_numpy(frame).float()
            with torch.no_grad():
                prob = silero_vad(tensor, 16000).item()

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
        command_queue.put("stop")
        llm_queue.put(("stop", ""))
        tts_engine.stop()
        
        print(f"\nPipeline stopped. Elapsed: {interviewer.elapsed_minutes():.1f} min")
        print("\nFull Q&A History:")
        for i, qa in enumerate(sm.qa_history):
            print(f"\nQ{i+1}: {qa['question']}")
            print(f"A:   {qa['answer']}")
            print(f"     Duration: {qa['duration_sec']}s | Pauses: {qa['pause_count']}")


if __name__ == "__main__":
    mock_resume = {"education": "B.Tech CS"}
    run_pipeline(resume_parsed=mock_resume)
