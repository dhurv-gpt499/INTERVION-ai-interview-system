import pyaudio
import queue
import threading
import numpy as np


class AudioCapture:
    def __init__(
        self,
        device_index=None,
        sample_rate=16000,
        chunk_size=1600,
        channels=1,
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels

        self.audio_queue = queue.Queue(maxsize=50)
        self.running = False
        self.p = pyaudio.PyAudio()
        self.stream = None

    @staticmethod
    def get_devices():
        p = pyaudio.PyAudio()
        devices = []
        default_index = -1
        try:
            default_device = p.get_default_input_device_info()
            default_index = default_device['index']
        except IOError:
            pass

        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                devices.append({
                    "index": i,
                    "name": info.get('name'),
                    "is_default": i == default_index
                })
        p.terminate()
        return devices

    def bytes_to_float32(self, audio_bytes):
        return (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

    def _capture_loop(self):
        while self.running:
            try:
                chunk = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
            except (IOError, OSError, Exception) as e:
                print(f"Error reading from mic: {e}")
                self.running = False
                break
                
            audio_np = self.bytes_to_float32(chunk)

            try:
                self.audio_queue.put_nowait(audio_np)
            except queue.Full:
                pass

    def start(self):
        if self.running:
            return

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
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
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass

        self.p.terminate()

    def get_chunk(self):
        while self.running:
            try:
                return self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
        return None