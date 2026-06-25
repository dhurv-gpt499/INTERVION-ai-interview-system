import numpy as np


class SpeechSegmenter:
    def __init__(
        self,
        speech_queue,
        on_event,
        silence_frames_threshold = 94,   # 3s  = 94  × 32ms
        no_answer_threshold      = 125,  # 4s  = 125 × 32ms
        max_speech_frames        = 250,  # 8s  = 250 × 32ms
        vad_threshold            = 0.5,
    ):
        self.speech_queue             = speech_queue
        self.on_event                 = on_event

        self.vad_threshold            = vad_threshold
        self.silence_frames_threshold = silence_frames_threshold
        self.no_answer_threshold      = no_answer_threshold
        self.max_speech_frames        = max_speech_frames

        self.speech_buffer            = []
        self.silence_frames           = 0
        self.speech_frames            = 0
        self.answer_started           = False
        self.no_answer_frames         = 0
        self.consecutive_speech       = 0    # ← NEW: debounce noise

    def process_frame(self, frame, speech_probability):
        is_speech = speech_probability > self.vad_threshold

        if is_speech:
            self.answer_started    = True
            self.no_answer_frames  = 0
            self.speech_buffer.append(frame)
            self.speech_frames    += 1
            self.consecutive_speech += 1     # ← NEW: count consecutive

            # only reset silence counter on SUSTAINED speech (not noise blip)
            if self.consecutive_speech >= 3: # ← NEW: 3 frames = ~96ms
                self.silence_frames = 0

            if self.speech_frames >= self.max_speech_frames:
                self._finalize("force_chunk")

        else:
            self.consecutive_speech = 0      # ← NEW: reset on silence

            if not self.answer_started:
                self.no_answer_frames += 1
                if self.no_answer_frames >= self.no_answer_threshold:
                    self.no_answer_frames = 0
                    self.on_event("no_answer_silence")

            if self.speech_buffer:
                self.silence_frames += 1
                self.speech_buffer.append(frame)

                if self.silence_frames >= self.silence_frames_threshold:
                    self._finalize("post_speech_silence")

    def _finalize(self, event_type):
        if not self.speech_buffer:
            return

        audio = np.concatenate(self.speech_buffer)
        self.speech_queue.put(audio)

        self.speech_buffer.clear()
        self.speech_frames  = 0
        self.silence_frames = 0

        self.on_event(event_type)

    def reset(self):
        """Call at the start of every new question."""
        self.speech_buffer.clear()
        self.speech_frames      = 0
        self.silence_frames     = 0
        self.answer_started     = False
        self.no_answer_frames   = 0
        self.consecutive_speech = 0    # ← NEW