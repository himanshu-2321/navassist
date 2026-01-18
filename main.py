import cv2
import pyttsx3
import time
import threading
import queue
import pygame
import os
from ultralytics import YOLO
import config  # Imports your config.py settings

class NavAssistCore:
    def __init__(self):
        # --- AI Configuration ---
        print("Loading AI Model...")
        self.model = YOLO('best.pt')  # Uses your trained model
        self.focal_length = 600
        
        # --- Audio Configuration ---
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # --- Sound Effect Setup (Pygame) ---
        pygame.mixer.init()
        self.sound_file = "beep-beep-6151.mp3"
        self.alert_sound = None
        
        if os.path.exists(self.sound_file):
            try:
                self.alert_sound = pygame.mixer.Sound(self.sound_file)
                print("✓ Alert sound loaded")
            except Exception as e:
                print(f"⚠ Audio error: {e}")
        else:
            print("⚠ Warning: Beep file not found.")

        # --- Thread-Safe Audio Queue ---
        # Prevents "Run loop already started" errors
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.last_speech_time = 0
        self.speech_cooldown = 3.0
        
        # Start the background audio worker
        threading.Thread(target=self.speech_worker, daemon=True).start()

    def speech_worker(self):
        """Background thread that handles talking and beeping"""
        while self.is_running:
            try:
                item = self.speech_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if item is None: break

            text, is_urgent = item
            
            try:
                # 1. Play Beep (If Urgent)
                if is_urgent and self.alert_sound:
                    self.alert_sound.play()
                    time.sleep(0.5)

                # 2. Speak Text
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech Error: {e}")
            
            self.speech_queue.task_done()

    def speak_warning(self, text, priority_level):
        """Adds text to the queue based on priority"""
        current_time = time.time()
        is_urgent = (priority_level >= 4)
        
        # Priority 4 (Critical) interrupts everything
        if is_urgent:
            try:
                self.engine.stop()
                with self.speech_queue.mutex:
                    self.speech_queue.queue.clear()
            except:
                pass
            self.speech_queue.put((text, True))
            self.last_speech_time = current_time
            
        # Standard Priority (only if cooldown passed)
        elif current_time - self.last_speech_time > self.speech_cooldown:
            self.speech_queue.put((text, False))
            self.last_speech_time = current_time

    def get_direction(self, x1, x2, width):
        """Returns: Left / Right / Ahead"""
        center = (x1 + x2) / 2
        if center < width / 3: return "left"
        if center > (width / 3) * 2: return "right"
        return "ahead"

    def get_audio_phrase(self, class_name, dist, direction, group):
        """Returns the specific text from your documentation"""
        if group == 'G1': # Critical
            return f"Warning! {class_name} detected {dist} meters {direction}. Stop."
        if group == 'G6' and dist < 3: # Traffic
            return f"Stop! {class_name} approaching {direction}."
        if group == 'G3': # Construction
            return f"Caution, construction hazard {dist} meters {direction}."
        return f"{class_name} {dist} meters {direction}."

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        
        print("System Started. Press 'Q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            height, width, _ = frame.shape
            
            # Run YOLO
            results = self.model(frame, stream=True, conf=0.4, verbose=False)
            
            highest_priority = 0
            audio_message = ""

            for r in results:
                for box in r.boxes:
                    # Data Extraction
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    # Distance Logic
                    real_h = config.OBJECT_HEIGHTS.get(class_name, 1.0)
                    bbox_h = y2 - y1
                    dist = round((real_h * self.focal_length) / bbox_h, 1) if bbox_h > 0 else 0
                    
                    # Context
                    group = config.CLASS_TO_GROUP.get(class_name, 'G8')
                    direction = self.get_direction(x1, x2, width)
                    
                    # Risk Scoring (The logic from your document)
                    risk = 1
                    if group == 'G1': risk = 4          # Fire/Weapons
                    elif group == 'G6' and dist < 4: risk = 4  # Close Traffic
                    elif group == 'G3' and dist < 2: risk = 3  # Construction
                    elif dist < 1.0: risk = 3           # Close Obstacles
                    elif group in ['G4', 'G7']: risk = 2
                    
                    # Visualization Colors
                    color = (0, 255, 0) # Green
                    if risk == 4: color = (0, 0, 255) # Red
                    elif risk == 3: color = (0, 165, 255) # Orange
                    
                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} {dist}m", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Prioritize Audio
                    if risk > highest_priority:
                        highest_priority = risk
                        audio_message = self.get_audio_phrase(class_name, dist, direction, group)

            # Trigger Audio if needed
            if highest_priority > 0 and audio_message:
                self.speak_warning(audio_message, highest_priority)
                # Visual Alert on Screen
                if highest_priority >= 3:
                    cv2.putText(frame, f"WARNING: {audio_message}", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            cv2.imshow("NavAssist Headless Mode", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.is_running = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = NavAssistCore()
    app.run()
