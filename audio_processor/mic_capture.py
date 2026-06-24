import pyaudio
import queue
import threading
import numpy as np


class AudioCapture:
    def __init__(
        self,
        sample_rate=16000,
        chunk_size=1600,
        channels=1,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels

        self.audio_queue = queue.Queue(maxsize=50)  # ← fix 1: size limit
        self.running = False
        self.p = pyaudio.PyAudio()
        self.stream = None  # ← stream starts as None

    def bytes_to_float32(self, audio_bytes):
        return (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

    def _capture_loop(self):
        while self.running:
            chunk = self.stream.read(
                self.chunk_size,
                exception_on_overflow=False,
            )
            audio_np = self.bytes_to_float32(chunk)

            try:
                self.audio_queue.put_nowait(audio_np)  # ← fix 2: non-blocking
            except queue.Full:
                pass  # drop frame if queue full, keep moving

    def start(self):
        if self.running:
            return

        # ← fix 1: stream opens HERE not in __init__
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if hasattr(self, "thread"):
            self.thread.join(timeout=1)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.p.terminate()

    def get_chunk(self):
        return self.audio_queue.get()


if __name__ == "__main__":
    audio = AudioCapture()
    audio.start()

    try:
        while True:
            chunk = audio.get_chunk()
            print(
                f"Shape={chunk.shape} "
                f"Min={chunk.min():.3f} "
                f"Max={chunk.max():.3f}"
            )
    except KeyboardInterrupt:
        audio.stop()