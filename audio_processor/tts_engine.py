import asyncio
import edge_tts
import pygame
import io
import queue
import threading

VOICE = "en-US-AndrewNeural"
pygame.mixer.init()

# Queue 1: text sentences waiting to be generated
_text_queue  = queue.Queue()
# Queue 2: pre-generated audio ready to play
_audio_queue = queue.Queue(maxsize=3)

_loop = asyncio.new_event_loop()

# Stop flag — when set, workers skip processing and unblock joins
_stop_event = threading.Event()

# Fires exactly when pygame starts playing the FIRST sentence of a stream
# This is the correct moment to switch the avatar to "talking"
_on_play_start = [None]  # list so inner scope can mutate it


# ── Generator thread: text → audio bytes ───────────────────────────────
async def _generate_audio(text: str) -> bytes:
    communicate  = edge_tts.Communicate(text, VOICE)
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    audio_buffer.seek(0)
    return audio_buffer


def _generator_worker():
    asyncio.set_event_loop(_loop)
    while True:
        text = _text_queue.get()
        if text is None:
            _audio_queue.put(None)
            _text_queue.task_done()
            break
        if _stop_event.is_set():
            # Stopped mid-flight: mark done and discard
            _text_queue.task_done()
            continue
        audio = _loop.run_until_complete(_generate_audio(text))
        if not _stop_event.is_set():
            _audio_queue.put((text, audio))
        _text_queue.task_done()


# ── Player thread: plays pre-generated audio immediately ────────────────
def _player_worker():
    while True:
        item = _audio_queue.get()
        if item is None:
            _audio_queue.task_done()
            break
        if _stop_event.is_set():
            # Stopped mid-flight: discard without playing
            _audio_queue.task_done()
            continue
        text, audio = item
        pygame.mixer.music.load(audio)
        pygame.mixer.music.play()

        # ← Avatar switch happens HERE — audio is actually playing now
        cb = _on_play_start[0]
        if cb:
            cb()
            _on_play_start[0] = None  # only fire once per stream session

        while pygame.mixer.music.get_busy():
            if _stop_event.is_set():
                pygame.mixer.music.stop()
                break
            pygame.time.wait(10)
        _audio_queue.task_done()


# Start both workers
_gen_thread    = threading.Thread(target=_generator_worker, daemon=True)
_player_thread = threading.Thread(target=_player_worker,    daemon=True)
_gen_thread.start()
_player_thread.start()


# ── Sentence extractor ──────────────────────────────────────────────────
def _extract_sentences(buffer: str) -> tuple:
    sentences, current = [], ""
    for char in buffer:
        current += char
        if char in ".!?" and current.strip():
            sentences.append(current.strip())
            current = ""
    return sentences, current


# ── Public API ──────────────────────────────────────────────────────────
def stream_from_llm(token_generator, state_machine=None, on_sentence=None, on_audio_start=None):
    """
    Streams LLM tokens → TTS → audio playback.

    on_sentence    : fires when a sentence is extracted from the token stream
    on_audio_start : fires the FIRST time pygame actually starts playing audio
                     Use this to switch the avatar to "talking" with 0 false positives
    """
    sentence_buffer = ""
    if state_machine:
        state_machine.is_speaking = True

    # Register the play-start callback for this stream session
    _on_play_start[0] = on_audio_start

    for token in token_generator:
        if _stop_event.is_set():
            break
        sentence_buffer += token
        sentences, sentence_buffer = _extract_sentences(sentence_buffer)
        for sentence in sentences:
            if sentence:
                print(f"[TTS] -> {sentence}")
                if on_sentence:
                    on_sentence(sentence)
                _text_queue.put(sentence)

    if sentence_buffer.strip() and not _stop_event.is_set():
        sent = sentence_buffer.strip()
        if on_sentence:
            on_sentence(sent)
        _text_queue.put(sent)

    # Wait for all audio to finish playing
    _text_queue.join()
    _audio_queue.join()

    if state_machine:
        state_machine.is_speaking = False


def speak_sync(text: str):
    _text_queue.put(text)
    _text_queue.join()
    _audio_queue.join()


def stop():
    """Immediately stops all audio and flushes queues. Safe to call any time."""
    _stop_event.set()
    _on_play_start[0] = None

    # Stop playback right now
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    # Drain queued items so join() unblocks
    for q in (_text_queue, _audio_queue):
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break

    # Clear stop flag after 1s so next interview session works normally
    threading.Timer(1.0, _stop_event.clear).start()