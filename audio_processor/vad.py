import numpy as np


class SpeechSegmenter:
    def __init__(
        self,
        speech_queue,
        on_event,                        # callback: receives event name
        silence_frames_threshold=30,     # 3s  = 30 frames (post speech)
        no_answer_threshold=40,          # 4s  = 40 frames (no answer)
        max_speech_frames=80,            # 8s  = 80 frames (force chunk)
        vad_threshold=0.5,
    ):
        self.speech_queue = speech_queue
        self.on_event = on_event         # fires: "no_answer_silence"
                                         #        "post_speech_silence"
                                         #        "force_chunk"

        self.vad_threshold = vad_threshold
        self.silence_frames_threshold = silence_frames_threshold
        self.no_answer_threshold = no_answer_threshold
        self.max_speech_frames = max_speech_frames

        self.speech_buffer = []
        self.silence_frames = 0
        self.speech_frames = 0

        self.answer_started = False       # has candidate spoken at all?
        self.no_answer_frames = 0         # silence counter before speech starts

    def process_frame(self, frame, speech_probability):
        is_speech = speech_probability > self.vad_threshold

        if is_speech:
            self.answer_started = True
            self.no_answer_frames = 0     # reset no-answer counter

            self.speech_buffer.append(frame)
            self.speech_frames += 1
            self.silence_frames = 0

            # force finalize if speech too long
            if self.speech_frames >= self.max_speech_frames:
                self._finalize("force_chunk")

        else:
            # candidate hasn't spoken yet → track for "no answer" event
            if not self.answer_started:
                self.no_answer_frames += 1

                if self.no_answer_frames >= self.no_answer_threshold:
                    self.no_answer_frames = 0   # reset to avoid repeated firing
                    self.on_event("no_answer_silence")

            # candidate was speaking → track post-speech silence
            if self.speech_buffer:
                self.silence_frames += 1
                self.speech_buffer.append(frame)  # keep silence in buffer

                if self.silence_frames >= self.silence_frames_threshold:
                    self._finalize("post_speech_silence")

    def _finalize(self, event_type):
        if not self.speech_buffer:
            return

        audio = np.concatenate(self.speech_buffer)
        self.speech_queue.put(audio)

        # reset state
        self.speech_buffer.clear()
        self.speech_frames = 0
        self.silence_frames = 0

        # fire event AFTER putting audio in queue
        self.on_event(event_type)

    def reset(self):
        """Call this at the start of every new question."""
        self.speech_buffer.clear()
        self.speech_frames = 0
        self.silence_frames = 0
        self.answer_started = False
        self.no_answer_frames = 0