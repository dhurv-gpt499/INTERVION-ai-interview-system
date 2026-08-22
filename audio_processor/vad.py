import numpy as np


class SpeechSegmenter:
    def __init__(
        self,
        speech_queue,
        on_event,
        silence_frames_threshold = 150,   # 5s  — post speech silence before finalizing (longer pause for thinking)
        no_answer_threshold      = 469,   # 15s — before first word (comfortable thinking time)
        max_speech_frames        = 250,   # 8s  — force chunk
        vad_threshold            = 0.3,   # Lowered from 0.5 to catch quieter mics
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

    def process_frame(self, frame, speech_probability):
        is_speech = speech_probability > self.vad_threshold

        if is_speech:
            self.answer_started    = True
            self.no_answer_frames  = 0
            self.speech_buffer.append(frame)
            self.speech_frames    += 1
            
            # Reset silence immediately on any speech frame to prevent fragmented cutoffs
            self.silence_frames = 0

            if self.speech_frames >= self.max_speech_frames:
                self._finalize("force_chunk")

        else:
            if not self.answer_started:
                self.no_answer_frames += 1
                if self.no_answer_frames >= self.no_answer_threshold:
                    self.no_answer_frames = 0
                    self.on_event("no_answer_silence")

            if self.speech_buffer:
                self.silence_frames += 1
                # Do NOT append silence frames to speech buffer to avoid Whisper hallucinations on room noise

                if self.silence_frames >= self.silence_frames_threshold:
                    self._finalize("post_speech_silence")

    def _finalize(self, event_type):
        if not self.speech_buffer:
            return

        if self.speech_queue is not None:
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