import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices):
    # compute distances between vertical eye landmarks
    p2_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    p3_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    # compute distance between horizontal eye landmarks
    p1_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

class AnxietyDetector:
    def __init__(self):
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.is_blinking = False
        
        self.prev_nose_pos = None
        self.fidget_score = 0.0
        
        self.anxiety_score = 10.0
        self.confidence_score = 90.0

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            # Face not found: increases anxiety score (looking away/hiding)
            self.anxiety_score = min(100.0, self.anxiety_score + 1.0)
            self.confidence_score = max(0.0, self.confidence_score - 1.0)
            return self.anxiety_score, self.confidence_score, rgb_frame
            
        landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array for easier math
        h, w, _ = frame.shape
        points = np.array([[p.x * w, p.y * h] for p in landmarks.landmark])
        
        # 1. Blink Detection
        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < 0.22: # threshold for closed eye
            if not self.is_blinking:
                self.blink_count += 1
                self.is_blinking = True
                self.last_blink_time = time.time()
        else:
            self.is_blinking = False
            
        # 2. Fidget / Head Movement Detection
        nose_tip = points[1]
        if self.prev_nose_pos is not None:
            movement = np.linalg.norm(nose_tip - self.prev_nose_pos)
            # Add small movements to fidget score
            if movement > 2.0:
                self.fidget_score += movement * 0.1
        self.prev_nose_pos = nose_tip
        
        # Decay fidget score over time
        self.fidget_score = max(0.0, self.fidget_score - 0.5)
        
        # Blink rate (blinks per minute approx based on last 10 seconds)
        # For simplicity, we just add anxiety per blink and decay it
        time_since_blink = time.time() - self.last_blink_time
        blink_anxiety = 0
        if time_since_blink < 1.0:
            blink_anxiety = 15.0 # bump anxiety momentarily on rapid blinks
            
        # Combine metrics: Fidget + Blink
        raw_anxiety = min(100.0, (self.fidget_score * 2.0) + blink_anxiety)
        
        # Smooth the scores (EMA)
        self.anxiety_score = self.anxiety_score * 0.95 + raw_anxiety * 0.05
        self.confidence_score = 100.0 - self.anxiety_score
        
        # Draw on frame for debug/display if needed
        cv2.putText(rgb_frame, f"Anxiety: {int(self.anxiety_score)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(rgb_frame, f"Conf: {int(self.confidence_score)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return self.anxiety_score, self.confidence_score, rgb_frame
