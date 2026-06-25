import edge_tts
import asyncio
import pygame
import io
import queue
import threading

VOICE = "en-US-AriaNeural"
pygame.mixer.init()

# ─── Single persistent queue + event loop for TTS ─────────────────────
_tts_queue = queue.Queue()
_loop       = asyncio.new_event_loop()


async def _generate_and_play(text: str):
    """Generate audio and play — runs inside persistent loop."""
    communicate  = edge_tts.Communicate(text, VOICE)
    audio_buffer = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    audio_buffer.seek(0)
    pygame.mixer.music.load(audio_buffer)
    pygame.mixer.music.play()

    # async wait — doesn't block the event loop
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.01)


def _tts_worker():
    """Dedicated background thread — one event loop, lives forever."""
    asyncio.set_event_loop(_loop)
    while True:
        text = _tts_queue.get()
        if text is None:            # shutdown signal
            break
        _loop.run_until_complete(_generate_and_play(text))
        _tts_queue.task_done()


# start worker thread once at import time
_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


# ─── Sentence extractor ────────────────────────────────────────────────
def _extract_sentences(buffer: str) -> tuple:
    """
    Scan char by char — extract complete sentences.
    Returns (list of complete sentences, leftover buffer)
    """
    sentences = []
    current   = ""

    for char in buffer:
        current += char
        if char in ".!?" and current.strip():
            sentences.append(current.strip())
            current = ""

    return sentences, current


# ─── Public API ────────────────────────────────────────────────────────
def stream_from_llm(token_generator, state_machine=None):
    """
    Feed LLM token stream directly into TTS.

    LLM keeps generating on THIS thread.
    TTS plays on BACKGROUND thread via queue.
    No blocking between generation and playback.
    """
    sentence_buffer = ""

    if state_machine:
        state_machine.is_speaking = True

    for token in token_generator:
        sentence_buffer += token

        sentences, sentence_buffer = _extract_sentences(sentence_buffer)

        for sentence in sentences:
            if sentence:
                print(f"[TTS] -> {sentence}")
                _tts_queue.put(sentence)   # non-blocking

    # any leftover text without punctuation
    if sentence_buffer.strip():
        _tts_queue.put(sentence_buffer.strip())

    # wait for all queued sentences to finish playing
    _tts_queue.join()

    if state_machine:
        state_machine.is_speaking = False


def speak_sync(text: str):
    """Speak a single piece of text directly."""
    _tts_queue.put(text)
    _tts_queue.join()


def stop():
    """Stop playback immediately — call when candidate interrupts."""
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    # drain the queue
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
            _tts_queue.task_done()
        except queue.Empty:
            break