import asyncio
import json
import torch
import numpy as np
import base64
from fastapi import WebSocket, WebSocketDisconnect

# Import existing logic
from audio_processor.model_loader import load_models
from audio_processor.vad import SpeechSegmenter
from audio_processor.interview_state_machine import InterviewStateMachine, InterviewState
from llm_interviewer.interviewer import QwenInterviewer
from llm_interviewer.answer_evaluator import AnswerEvaluator

from v2.server.stt_engine import STTEngine
import edge_tts

async def handle_interview_session(websocket: WebSocket, config: dict):
    # Extract config
    resume_parsed = config.get("resume_parsed", {})
    companies = config.get("companies", ["Google"])
    roles = config.get("roles", ["Software Engineer"])
    level = config.get("level", "Mid-Level")
    llm_backend = config.get("llm_backend", "Cloud API (Groq)")
    api_key = config.get("llm_api_key", "")
    use_groq_stt = "Groq" in llm_backend  # Use Groq STT if using Groq LLM
    
    # Initialize components
    sm = InterviewStateMachine()
    interviewer = QwenInterviewer(llm_backend=llm_backend, api_key=api_key)
    evaluator = AnswerEvaluator(llm_backend=llm_backend, api_key=api_key)
    stt_engine = STTEngine(use_groq=use_groq_stt, api_key=api_key)
    
    # We still need silero VAD
    _, silero_vad = load_models()
    
    # Event queues for communication between ws listener and processing task
    speech_queue = asyncio.Queue()
    command_queue = asyncio.Queue()
    
    # Setup VAD segmenter
    def on_vad_event(event_type):
        # We need to dispatch to the async loop safely
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(command_queue.put_nowait, event_type)
        
    segmenter = SpeechSegmenter(speech_queue=None, on_event=on_vad_event, silence_frames_threshold=150)
    
    # Setup state notification helper
    async def notify_state(state: str, payload: dict = None):
        msg = {"type": "state", "state": state}
        if payload:
            msg.update(payload)
        await websocket.send_json(msg)
        
    async def notify_transcript(text: str, is_final: bool = False):
        if text:
            await websocket.send_json({"type": "transcript", "text": text, "is_final": is_final})
            
    # Send initial setup
    await notify_state("loading")
    interviewer.build_system_prompt(
        resume_parsed=resume_parsed,
        preferred_companies=companies,
        preferred_roles=roles,
        target_level=level
    )
    
    # Generate first question
    sm.transition(InterviewState.AI_SPEAKING)
    await notify_state("ai_speaking")
    
    first_response = interviewer.receive_answer("")
    await stream_tts_to_websocket(first_response, websocket, notify_transcript)
    
    sm.set_question(interviewer.messages[-1]["content"])
    sm.transition(InterviewState.QUESTION_ASKED)
    await notify_state("question_asked")
    
    # ---------------------------------------------------------
    # Task 1: Listen to WebSocket for Audio Blobs
    # ---------------------------------------------------------
    async def ws_listener():
        try:
            while True:
                data = await websocket.receive()
                if "bytes" in data:
                    # Client sends 16kHz float32 or int16 PCM
                    audio_bytes = data["bytes"]
                    # Convert to numpy array float32
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    if sm.current_state in (InterviewState.QUESTION_ASKED, InterviewState.LISTENING):
                        # Run VAD (blocking call but extremely fast, ok for main loop)
                        tensor = torch.from_numpy(audio_np).float()
                        with torch.no_grad():
                            prob = silero_vad(tensor, 16000).item()
                        
                        is_speech = prob > segmenter.vad_threshold
                        if is_speech and sm.current_state == InterviewState.QUESTION_ASKED:
                            sm.start_answer()
                            sm.transition(InterviewState.LISTENING)
                            await notify_state("listening")
                            
                        segmenter.process_frame(audio_np, prob)
                        
                elif "text" in data:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "stop":
                        await command_queue.put("stop")
        except WebSocketDisconnect:
            await command_queue.put("stop")
            
    # ---------------------------------------------------------
    # Task 2: Process Commands (VAD Events & LLM)
    # ---------------------------------------------------------
    async def command_processor():
        while True:
            cmd = await command_queue.get()
            if cmd == "stop":
                break
                
            elif cmd == "post_speech_silence":
                if not segmenter.speech_buffer:
                    continue
                    
                # Concatenate all frames in buffer
                full_audio = np.concatenate(segmenter.speech_buffer)
                segmenter.speech_buffer.clear()
                
                await notify_state("evaluating")
                # STT
                text = await stt_engine.transcribe_async(full_audio)
                sm.append_transcript(text)
                await notify_transcript(text, is_final=True)
                
                sm.finalize_answer()
                sm.transition(InterviewState.ANSWER_COMPLETE)
                
                # LLM & TTS
                response_generator = interviewer.receive_answer(sm.final_answer)
                
                sm.transition(InterviewState.AI_SPEAKING)
                await notify_state("ai_speaking")
                
                await stream_tts_to_websocket(response_generator, websocket, notify_transcript)
                
                if not interviewer.is_active:
                    sm.transition(InterviewState.SESSION_COMPLETE)
                    await notify_state("session_complete")
                    
                    # Run Evaluator
                    report = evaluator.evaluate_final_interview(
                        qa_history=sm.qa_history,
                        resume_parsed=resume_parsed,
                        preferred_companies=companies,
                        target_level=level,
                        avg_anxiety=50, # Dummy for now
                        avg_confidence=50
                    )
                    await websocket.send_json({"type": "report", "report": report})
                    break
                else:
                    sm.set_question(interviewer.messages[-1]["content"])
                    sm.save_to_history()
                    segmenter.reset()
                    sm.transition(InterviewState.QUESTION_ASKED)
                    await notify_state("question_asked")
                    
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
                await stream_tts_to_websocket(response_generator, websocket, notify_transcript)
                segmenter.reset()
                sm.transition(InterviewState.QUESTION_ASKED)
                await notify_state("question_asked")
                
    # Run tasks concurrently
    listener_task = asyncio.create_task(ws_listener())
    processor_task = asyncio.create_task(command_processor())
    
    # Wait for either to finish (e.g. client disconnects or interview ends)
    done, pending = await asyncio.wait(
        [listener_task, processor_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    for p in pending:
        p.cancel()

async def stream_tts_to_websocket(response_generator, websocket: WebSocket, notify_transcript):
    """
    Consumes LLM generator, sends transcript text, and generates Edge-TTS audio.
    Sends raw audio bytes to the client.
    """
    import re
    full_text = ""
    for chunk in response_generator:
        full_text += chunk
        
    await notify_transcript(f"??: {full_text}", is_final=True)
    
    # Generate TTS
    communicate = edge_tts.Communicate(full_text, "en-US-ChristopherNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            # Send binary audio chunk
            await websocket.send_bytes(chunk["data"])
    
    # Send a marker that TTS is done
    await websocket.send_json({"type": "tts_complete"})

