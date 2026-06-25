import threading
import time
from enum import Enum


class InterviewState(Enum):
    IDLE             = "idle"
    QUESTION_ASKED   = "question_asked"
    LISTENING        = "listening"
    CANDIDATE_PAUSED = "candidate_paused"
    ANSWER_COMPLETE  = "answer_complete"
    NO_ANSWER        = "no_answer"
    EVALUATING       = "evaluating"
    AI_SPEAKING      = "ai_speaking"
    SESSION_COMPLETE = "session_complete"


class InterviewStateMachine:
    """
    Central state manager for the interview pipeline.
    All threads check and update state through here.
    threading.Lock() ensures no two threads clash.
    """

    # Valid transitions — what state can go to what
    VALID_TRANSITIONS = {
        InterviewState.IDLE             : [InterviewState.QUESTION_ASKED],
        InterviewState.QUESTION_ASKED   : [InterviewState.LISTENING,
                                           InterviewState.NO_ANSWER],
        InterviewState.LISTENING        : [InterviewState.CANDIDATE_PAUSED,
                                           InterviewState.ANSWER_COMPLETE,
                                           InterviewState.EVALUATING],
        InterviewState.CANDIDATE_PAUSED : [InterviewState.LISTENING,
                                           InterviewState.ANSWER_COMPLETE],
        InterviewState.ANSWER_COMPLETE  : [InterviewState.EVALUATING],
        InterviewState.NO_ANSWER        : [InterviewState.AI_SPEAKING,
                                           InterviewState.LISTENING],
        InterviewState.EVALUATING       : [InterviewState.AI_SPEAKING],
        InterviewState.AI_SPEAKING      : [InterviewState.QUESTION_ASKED,
                                           InterviewState.SESSION_COMPLETE],
        InterviewState.SESSION_COMPLETE : [],
    }

    def __init__(self):
        self.current_state  = InterviewState.IDLE
        self._lock          = threading.Lock()
        self._state_start   = time.time()

        # Interview data tracked alongside state
        self.question_number   = 0
        self.current_question  = ""
        self.partial_answer    = ""
        self.final_answer      = ""
        self.answer_start_time = None
        self.answer_duration   = 0.0
        self.pause_count       = 0
        self.qa_history        = []   # [{"question": ..., "answer": ...}]

    # ------------------------------------------------------------------
    # Core: transition to a new state
    # ------------------------------------------------------------------
    def transition(self, new_state: InterviewState) -> bool:
        with self._lock:
            allowed = self.VALID_TRANSITIONS.get(self.current_state, [])

            if new_state not in allowed:
                print(f"[STATE] Invalid: {self.current_state.value} → {new_state.value}")
                return False

            old = self.current_state.value
            self.current_state = new_state
            self._state_start  = time.time()
            print(f"[STATE] {old} → {new_state.value}")
            return True

    # ------------------------------------------------------------------
    # Convenience checks (no lock needed — just reading)
    # ------------------------------------------------------------------
    def is_listening(self)   -> bool:
        return self.current_state == InterviewState.LISTENING

    def is_ai_speaking(self) -> bool:
        return self.current_state == InterviewState.AI_SPEAKING

    def is_idle(self)        -> bool:
        return self.current_state == InterviewState.IDLE

    def time_in_state(self)  -> float:
        return time.time() - self._state_start

    # ------------------------------------------------------------------
    # Answer tracking (called by STT thread)
    # ------------------------------------------------------------------
    def start_answer(self):
        """Called when candidate starts speaking."""
        with self._lock:
            self.answer_start_time = time.time()
            self.partial_answer    = ""

    def append_transcript(self, text: str):
        """Called after each Whisper chunk — thread safe."""
        with self._lock:
            self.partial_answer += " " + text
            self.partial_answer  = self.partial_answer.strip()

    def finalize_answer(self):
        """Called on post_speech_silence — locks in the answer."""
        with self._lock:
            self.final_answer = self.partial_answer
            if self.answer_start_time:
                self.answer_duration = time.time() - self.answer_start_time

    # ------------------------------------------------------------------
    # Question management
    # ------------------------------------------------------------------
    def set_question(self, question: str):
        """Call before transitioning to QUESTION_ASKED."""
        with self._lock:
            self.current_question = question
            self.partial_answer   = ""
            self.final_answer     = ""
            self.answer_duration  = 0.0
            self.pause_count      = 0
            self.question_number += 1

    def save_to_history(self):
        """Call after answer is finalized."""
        with self._lock:
            self.qa_history.append({
                "question"        : self.current_question,
                "answer"          : self.final_answer,
                "duration_sec"    : round(self.answer_duration, 2),
                "pause_count"     : self.pause_count,
                "question_number" : self.question_number,
            })

    def increment_pause(self):
        with self._lock:
            self.pause_count += 1

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------
    def status(self):
        return {
            "state"           : self.current_state.value,
            "question"        : self.current_question,
            "partial_answer"  : self.partial_answer,
            "pause_count"     : self.pause_count,
            "time_in_state"   : round(self.time_in_state(), 1),
        }