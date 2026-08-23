import asyncio
import json
import torch
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
import re

# Import existing logic
from audio_processor.model_loader import load_models
from audio_processor.vad import SpeechSegmenter
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState
from llm_interviewer.interviewer import QwenInterviewer
from llm_interviewer.answer_evaluator import AnswerEvaluator
from server.stt_engine import STTEngine
from vision_processor.camera import VideoCaptureThread
import edge_tts


async def handle_interview_session(websocket: WebSocket, config: dict):
    # Extract config
    resume_parsed = config.get("resume_parsed", {})
    companies = config.get("companies", ["Google"])
    roles = config.get("roles", ["Software Engineer"])
    level = config.get("level", "Mid-Level")
    llm_backend = config.get("llm_backend", "Cloud API (Groq)")
    api_key = config.get("llm_api_key", "")
    use_groq_stt = "Groq" in llm_backend

    # Initialize components
    sm = InterviewStateMachine()
    interviewer = QwenInterviewer(llm_backend=llm_backend, api_key=api_key)
    evaluator = AnswerEvaluator(llm_backend=llm_backend, api_key=api_key)
    stt_engine = STTEngine(use_groq=use_groq_stt, api_key=api_key)

    # Load only Silero VAD (Whisper is handled by STTEngine)
    _, silero_vad = load_models()

    # Event queues
    command_queue = asyncio.Queue()
    tts_cancel_event = asyncio.Event()
    main_loop = asyncio.get_running_loop()

    # Setup VAD segmenter
    def on_vad_event(event_type):
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(command_queue.put_nowait, event_type)

    segmenter = SpeechSegmenter(speech_queue=None, on_event=on_vad_event)

    # Vision pipeline (webcam anxiety detection)
    vision_data = {"anxieties": [], "confidences": []}

    def on_vision_scores(anx, conf):
        vision_data["anxieties"].append(anx)
        vision_data["confidences"].append(conf)
        # We send these updates to the UI, but limit frequency to avoid overwhelming the WS
        if len(vision_data["anxieties"]) % 10 == 0:  # ~ 3 times per sec (30fps/10)
            main_loop.call_soon_threadsafe(command_queue.put_nowait, ("vision_update", anx, conf))

    vision_thread = VideoCaptureThread(src=0, on_scores=on_vision_scores)
    vision_thread.start()

    # --- Helper functions ---
    async def notify_state(state: str, payload: dict = None):
        try:
            msg = {"type": "state", "state": state}
            if payload:
                msg.update(payload)
            await websocket.send_json(msg)
        except Exception:
            pass

    async def notify_transcript(text: str, is_final: bool = False):
        try:
            if text:
                await websocket.send_json({"type": "transcript", "text": text, "is_final": is_final})
        except Exception:
            pass

    # --- Start interview ---
    await notify_state("loading")

    # 1. Generate the omnipotent attack plan based on resume (takes ~15-30s)
    from llm_interviewer.attack_planner import generate_attack_plan
    target_company = companies[0] if companies else "General"
    target_role = roles[0] if roles else "Software Engineer"
    
    attack_plan = await generate_attack_plan(
        resume_text=resume_parsed.get("raw_resume", ""),
        target_company=target_company,
        target_role=target_role,
        llm_backend=llm_backend,
        api_key=api_key
    )
    
    # 2. Inject attack plan into resume_parsed for RAG index building
    resume_parsed["attack_plan"] = attack_plan

    first_response = interviewer.start(
        resume_parsed=resume_parsed,
        preferred_companies=companies,
        preferred_roles=roles,
        target_level=level,
        domain="General Tech"
    )

    sm.transition(InterviewState.AI_SPEAKING)
    tts_cancel_event.clear()
    tts_state = {"spoken_text": ""}
    await stream_tts_to_websocket(first_response, websocket, notify_transcript, notify_state, tts_cancel_event, tts_state)

    sm.set_question(interviewer.messages[-1]["content"])
    # We stay in AI_SPEAKING until playback_complete is received

    # ---------------------------------------------------------
    # Task 1: Listen to WebSocket for Audio Blobs
    # ---------------------------------------------------------
    async def ws_listener():
        try:
            while True:
                data = await websocket.receive()
                if "bytes" in data:
                    audio_bytes = data["bytes"]
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

                    # Process audio in QUESTION_ASKED, LISTENING, or AI_SPEAKING states
                    if sm.current_state in (InterviewState.QUESTION_ASKED, InterviewState.LISTENING, InterviewState.AI_SPEAKING):
                        chunk_size = 512
                        for i in range(0, len(audio_np), chunk_size):
                            chunk = audio_np[i:i + chunk_size]
                            if len(chunk) < chunk_size:
                                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')

                            tensor = torch.from_numpy(chunk.copy()).float()
                            with torch.no_grad():
                                prob = silero_vad(tensor, 16000).item()

                            is_speech = prob > segmenter.vad_threshold

                            if is_speech:
                                if sm.current_state == InterviewState.QUESTION_ASKED:
                                    sm.start_answer()
                                    sm.transition(InterviewState.LISTENING)
                                    await notify_state("listening")
                                elif sm.current_state == InterviewState.AI_SPEAKING:
                                    # Require sustained speech (e.g., 15 frames = ~0.5s) to interrupt
                                    # We attach this counter to the segmenter for convenience
                                    if not hasattr(segmenter, "interrupt_frames"):
                                        segmenter.interrupt_frames = 0
                                    segmenter.interrupt_frames += 1
                                    
                                    if segmenter.interrupt_frames > 15:
                                        # Candidate is truly interrupting the AI
                                        tts_cancel_event.set()
                                        await command_queue.put("interrupt")
                                        segmenter.interrupt_frames = 0
                            else:
                                if sm.current_state == InterviewState.AI_SPEAKING:
                                    if hasattr(segmenter, "interrupt_frames"):
                                        segmenter.interrupt_frames = 0

                            # Only feed VAD segmenter when listening
                            if sm.current_state in (InterviewState.QUESTION_ASKED, InterviewState.LISTENING):
                                segmenter.process_frame(chunk, prob)

                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "stop":
                        await command_queue.put("stop")
                    elif msg.get("type") == "playback_complete":
                        await command_queue.put("playback_complete")
        except WebSocketDisconnect:
            await command_queue.put("stop")
        except Exception:
            await command_queue.put("stop")

    # ---------------------------------------------------------
    # Task 2: Process Commands (VAD Events & LLM)
    # ---------------------------------------------------------
    async def command_processor():
        while True:
            cmd = await command_queue.get()
            if cmd == "stop":
                await notify_state("evaluating")
                try:
                    # Save any in-progress question/answer to the history log
                    sm.save_to_history()
                    
                    avg_anx = sum(vision_data["anxieties"])/len(vision_data["anxieties"]) if vision_data["anxieties"] else 0
                    avg_conf = sum(vision_data["confidences"])/len(vision_data["confidences"]) if vision_data["confidences"] else 0
                    
                    # Run synchronously blocking LLM call in a background thread
                    loop = asyncio.get_running_loop()
                    report = await loop.run_in_executor(
                        None,
                        lambda: evaluator.evaluate_final_interview(
                            qa_history=sm.qa_history,
                            resume_parsed=resume_parsed,
                            preferred_companies=companies,
                            target_level=level,
                            avg_anxiety=avg_anx,
                            avg_confidence=avg_conf
                        )
                    )
                    
                    await websocket.send_json({"type": "report", "report": report})
                except Exception as e:
                    print(f"Error generating early report: {e}")
                break

            elif isinstance(cmd, tuple) and cmd[0] == "vision_update":
                _, anx, conf = cmd
                try:
                    await websocket.send_json({"type": "vision", "anxiety": anx, "confidence": conf})
                except:
                    pass

            elif cmd == "interrupt":
                try:
                    await websocket.send_json({"type": "interrupt"})
                except Exception:
                    pass
                sm.start_answer()
                sm.transition(InterviewState.LISTENING)
                await notify_state("listening")

            elif cmd == "post_speech_silence":
                if not segmenter.speech_buffer:
                    continue

                full_audio = np.concatenate(segmenter.speech_buffer)
                segmenter.speech_buffer.clear()

                await notify_state("evaluating")

                # STT
                text = await stt_engine.transcribe_async(full_audio)
                sm.append_transcript(text)
                await notify_transcript(text, is_final=True)

                sm.finalize_answer()

                # Save history BEFORE set_question overwrites it
                final_answer = sm.final_answer

                sm.transition(InterviewState.AI_SPEAKING)

                # LLM & TTS
                response_generator = interviewer.receive_answer(final_answer)
                tts_cancel_event.clear()
                tts_state = {"spoken_text": ""}
                await stream_tts_to_websocket(response_generator, websocket, notify_transcript, notify_state, tts_cancel_event, tts_state)

                if tts_cancel_event.is_set():
                    # Interrupted mid-sentence! Truncate history to avoid hallucination
                    interviewer.truncate_last_response(tts_state["spoken_text"])
                    # The interrupt handler already put us into LISTENING mode, so we just break out of this command
                    continue

                if not interviewer.is_active:
                    sm.transition(InterviewState.SESSION_COMPLETE)
                    await notify_state("session_complete")

                    # Run Evaluator without blocking event loop
                    avg_anx = sum(vision_data["anxieties"]) / len(vision_data["anxieties"]) if vision_data["anxieties"] else 50.0
                    avg_conf = sum(vision_data["confidences"]) / len(vision_data["confidences"]) if vision_data["confidences"] else 50.0
                    
                    loop = asyncio.get_running_loop()
                    report = await loop.run_in_executor(
                        None,
                        lambda: evaluator.evaluate_final_interview(
                            qa_history=sm.qa_history,
                            resume_parsed=resume_parsed,
                            preferred_companies=companies,
                            target_level=level,
                            avg_anxiety=avg_anx,
                            avg_confidence=avg_conf
                        )
                    )
                    
                    await websocket.send_json({"type": "report", "report": report})
                    break
                else:
                    # Save current Q&A to history, THEN set next question
                    sm.save_to_history()
                    sm.set_question(interviewer.messages[-1]["content"])
                    # Wait for playback_complete to transition to QUESTION_ASKED

            elif cmd == "force_chunk":
                if not segmenter.speech_buffer:
                    continue
                full_audio = np.concatenate(segmenter.speech_buffer)
                segmenter.speech_buffer.clear()
                text = await stt_engine.transcribe_async(full_audio)
                sm.append_transcript(text)
                await notify_transcript(text, is_final=False)

            elif cmd == "no_answer_silence":
                sm.transition(InterviewState.AI_SPEAKING)
                await notify_state("ai_speaking")
                response_generator = interviewer.receive_answer("...silence...")
                tts_cancel_event.clear()
                tts_state = {"spoken_text": ""}
                await stream_tts_to_websocket(response_generator, websocket, notify_transcript, notify_state, tts_cancel_event, tts_state)

                if tts_cancel_event.is_set():
                    interviewer.truncate_last_response(tts_state["spoken_text"])
                    continue
                # Wait for playback_complete to transition to QUESTION_ASKED

            elif cmd == "playback_complete":
                if sm.current_state == InterviewState.AI_SPEAKING:
                    segmenter.reset()
                    sm.transition(InterviewState.QUESTION_ASKED)
                    await notify_state("question_asked")

    # Run tasks concurrently
    listener_task = asyncio.create_task(ws_listener())
    processor_task = asyncio.create_task(command_processor())

    done, pending = await asyncio.wait(
        [listener_task, processor_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    for p in pending:
        p.cancel()
    vision_thread.stop()


async def stream_tts_to_websocket(response_generator, websocket: WebSocket, notify_transcript, notify_state, tts_cancel_event: asyncio.Event = None, tts_state: dict = None):
    if tts_state is None:
        tts_state = {"spoken_text": ""}
        
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()
    has_spoken = False

    def producer():
        try:
            for chunk in response_generator:
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, producer)

    full_text = ""
    current_sentence = ""

    async def run_tts_and_send(text):
        if not text.strip():
            return
        # Strip markdown symbols for smooth TTS speaking
        clean_text = re.sub(r'[*`#~_\[\]]+', '', text)
        clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
        if not clean_text.strip():
            return
            
        try:
            communicate = edge_tts.Communicate(clean_text, "en-US-ChristopherNeural")
            audio_data = bytearray()
            async for chunk in communicate.stream():
                if tts_cancel_event and tts_cancel_event.is_set():
                    break
                if chunk["type"] == "audio":
                    audio_data.extend(chunk["data"])
            
            if audio_data and not (tts_cancel_event and tts_cancel_event.is_set()):
                await websocket.send_bytes(bytes(audio_data))
                # Only add to spoken_text if we successfully sent it
                tts_state["spoken_text"] += text + " "
        except Exception as e:
            print(f"[TTS] Error: {e}")

    while True:
        if tts_cancel_event and tts_cancel_event.is_set():
            break

        chunk = await q.get()
        if chunk is None:
            break
        if isinstance(chunk, Exception):
            print(f"[STREAM] Error from LLM generator: {chunk}")
            break

        if chunk == "<THINKING>":
            await notify_state("evaluating")
            continue

        # First actual text chunk — notify UI that AI is now speaking
        if not has_spoken and chunk.strip():
            has_spoken = True
            await notify_state("ai_speaking")

        full_text += chunk
        current_sentence += chunk

        # Stream transcript chunk (use "AI:" prefix)
        await notify_transcript(f"AI: {full_text}", is_final=False)

        # Flush on sentence boundaries to preserve natural prosody
        if any(p in chunk for p in ".!?\n"):
            await run_tts_and_send(current_sentence)
            current_sentence = ""

    if current_sentence and not (tts_cancel_event and tts_cancel_event.is_set()):
        await run_tts_and_send(current_sentence)

    await notify_transcript(f"AI: {full_text}", is_final=True)
    try:
        await websocket.send_json({"type": "tts_complete"})
    except Exception:
        pass
