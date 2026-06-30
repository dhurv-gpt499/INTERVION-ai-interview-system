import cv2
import threading
import time
from .anxiety_detector import AnxietyDetector

class VideoCaptureThread:
    def __init__(self, src=0, on_frame=None, on_scores=None):
        self.src = src
        self.cap = None
        self.running = False
        self.thread = None
        self.on_frame = on_frame
        self.on_scores = on_scores
        self.detector = AnxietyDetector()

    def start(self):
        self.cap = cv2.VideoCapture(self.src)
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Process frame through mediapipe
                anxiety, confidence, out_frame = self.detector.process_frame(frame)
                
                if self.on_scores:
                    self.on_scores(anxiety, confidence)
                
                if self.on_frame:
                    self.on_frame(out_frame)
            
            # small sleep to avoid maxing out CPU (30fps target)
            time.sleep(0.033)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
