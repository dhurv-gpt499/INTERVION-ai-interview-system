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
_audio_queue = queue.Queue(maxsize=3)   # buffer max 3 sentences ahead

_loop = asyncio.new_event_loop()


# ── Generator thread: text → audio bytes (runs ahead) ──────────────────
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
            break
        audio = _loop.run_until_complete(_generate_audio(text))
        _audio_queue.put((text, audio))   # push pre-generated audio
        _text_queue.task_done()


# ── Player thread: plays pre-generated audio immediately ───────────────
def _player_worker():
    while True:
        item = _audio_queue.get()
        if item is None:
            break
        text, audio = item
        pygame.mixer.music.load(audio)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(10)
        _audio_queue.task_done()


# Start both threads
_gen_thread    = threading.Thread(target=_generator_worker, daemon=True)
_player_thread = threading.Thread(target=_player_worker,    daemon=True)
_gen_thread.start()
_player_thread.start()


# ── Sentence extractor ─────────────────────────────────────────────────
def _extract_sentences(buffer: str) -> tuple:
    sentences, current = [], ""
    for char in buffer:
        current += char
        if char in ".!?" and current.strip():
            sentences.append(current.strip())
            current = ""
    return sentences, current


# ── Public API ─────────────────────────────────────────────────────────
def stream_from_llm(token_generator, state_machine=None, on_sentence=None):
    sentence_buffer = ""
    if state_machine:
        state_machine.is_speaking = True

    for token in token_generator:
        sentence_buffer += token
        sentences, sentence_buffer = _extract_sentences(sentence_buffer)
        for sentence in sentences:
            if sentence:
                print(f"[TTS] -> {sentence}")
                if on_sentence:
                    on_sentence(sentence)
                _text_queue.put(sentence)   # generator picks up immediately

    if sentence_buffer.strip():
        sent = sentence_buffer.strip()
        if on_sentence:
            on_sentence(sent)
        _text_queue.put(sent)

    # wait for all text to be generated AND played
    _text_queue.join()
    _audio_queue.join()

    if state_machine:
        state_machine.is_speaking = False


def speak_sync(text: str):
    _text_queue.put(text)
    _text_queue.join()
    _audio_queue.join()


def stop():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    for q in (_text_queue, _audio_queue):
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break