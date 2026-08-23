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
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                print("[VISION] Warning: Could not open webcam. Anxiety detection disabled.")
                self.cap = None
                return
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"[VISION] Warning: Webcam init failed: {e}. Anxiety detection disabled.")
            self.cap = None

    def _update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    anxiety, confidence, out_frame = self.detector.process_frame(frame)
                    
                    if self.on_scores:
                        self.on_scores(anxiety, confidence)
                    
                    if self.on_frame:
                        self.on_frame(out_frame)
            except Exception:
                pass
            
            # small sleep to avoid maxing out CPU (30fps target)
            time.sleep(0.033)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
