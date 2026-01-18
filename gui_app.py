import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import pyttsx3
import time
from ultralytics import YOLO
import config  # Importing your config.py
import queue
import pygame
import os

class NavAssistApp:
    # Color Scheme - Black & White / High Contrast
    COLORS = {
        'bg_dark': '#000000',      # Pure black
        'bg_light': '#1a1a1a',     # Dark gray
        'accent_blue': '#ffffff',  # White
        'accent_green': '#ffffff', # White
        'accent_red': '#ffffff',   # White
        'accent_orange': '#ffffff',# White
        'text_white': '#ffffff',   # White
        'text_gray': '#cccccc'     # Light gray
    }
    
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1280x800")
        self.window.configure(bg=self.COLORS['bg_dark'])
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('TLabel', background=self.COLORS['bg_dark'], foreground=self.COLORS['text_white'])
        
        # --- State Variables ---
        self.is_running = False
        self.is_speaking = False
        self.audio_enabled = True
        self.last_speech_time = 0
        self.speech_cooldown = 3.0  # Seconds between warnings
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()
        
        # --- AI & Camera Setup ---
        print("Loading Custom Model...")
        # Ensure best.pt is in the same directory or provide full path
        self.model = YOLO(r'best.pt')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.focal_length = 600
        
        # --- Audio Setup ---
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # --- Alert Sound Setup ---
        pygame.mixer.init()
        self.sound_file = "beep-beep-6151.mp3"
        self.alert_sound = None
        
        if os.path.exists(self.sound_file):
            try:
                self.alert_sound = pygame.mixer.Sound(self.sound_file)
                print(f"✓ Alert sound loaded")
            except Exception as e:
                print(f"⚠ Audio error: {e}")
        
        self.speech_queue = queue.Queue()
        threading.Thread(target=self.speech_worker, daemon=True).start()
        
        # --- GUI Layout ---
        # 1. Top Header
        self.create_header()
        
        # 2. Main Content (Video + Sidebar)
        self.create_main_content()
        
        # 3. Bottom Status Bar
        self.create_status_bar()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_video()
    
    def create_header(self):
        """Create professional header"""
        header = tk.Frame(self.window, bg=self.COLORS['bg_light'], height=70)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        
        # Border
        border = tk.Frame(self.window, bg=self.COLORS['text_white'], height=2)
        border.pack(side="top", fill="x")
        
        # Title
        title_label = tk.Label(header, text="NAVASSIST AI SAFETY SYSTEM", 
                               font=("Segoe UI", 16, "bold"),
                               bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        title_label.pack(side="left", padx=20, pady=15)
        
        # Control Buttons
        btn_frame = tk.Frame(header, bg=self.COLORS['bg_light'])
        btn_frame.pack(side="right", padx=20, pady=15)
        
        self.btn_start = tk.Button(btn_frame, text="START", bg=self.COLORS['text_white'], 
                                   fg=self.COLORS['bg_dark'], font=("Segoe UI", 10, "bold"),
                                   command=self.start_system, padx=15, pady=8, relief="flat")
        self.btn_start.pack(side="left", padx=5)
        
        self.btn_stop = tk.Button(btn_frame, text="STOP", bg=self.COLORS['text_white'],
                                  fg=self.COLORS['bg_dark'], font=("Segoe UI", 10, "bold"),
                                  command=self.stop_system, state="disabled", padx=15, pady=8, relief="flat")
        self.btn_stop.pack(side="left", padx=5)
        
        self.btn_audio = tk.Button(btn_frame, text="AUDIO", bg=self.COLORS['text_white'],
                                   fg=self.COLORS['bg_dark'], font=("Segoe UI", 10, "bold"),
                                   command=self.toggle_audio, padx=15, pady=8, relief="flat")
        self.btn_audio.pack(side="left", padx=5)
    
    def create_main_content(self):
        """Create video display + sidebar"""
        main_frame = tk.Frame(self.window, bg=self.COLORS['bg_dark'])
        main_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        # Video Display
        self.video_frame = tk.Label(main_frame, bg=self.COLORS['bg_light'], relief="sunken", bd=2)
        self.video_frame.pack(side="left", fill="both", expand=True)
        
        # Right Sidebar
        sidebar = tk.Frame(main_frame, bg=self.COLORS['bg_light'], width=250)
        sidebar.pack(side="right", fill="both", padx=(10, 0))
        sidebar.pack_propagate(False)
        
        # Separator line
        sep = tk.Frame(sidebar, bg=self.COLORS['text_white'], height=2)
        sep.pack(fill="x", pady=(10, 0))
        
        # Info Labels
        info_label = tk.Label(sidebar, text="DETECTION INFO", font=("Segoe UI", 11, "bold"),
                             bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        info_label.pack(pady=(15, 10), padx=10)
        
        # FPS
        self.fps_label = tk.Label(sidebar, text="FPS: 0", font=("Segoe UI", 10),
                                 bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        self.fps_label.pack(pady=5, padx=10, anchor="w")
        
        # Objects Detected
        self.obj_label = tk.Label(sidebar, text="Objects: 0", font=("Segoe UI", 10),
                                 bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        self.obj_label.pack(pady=5, padx=10, anchor="w")
        
        # Separator line
        sep2 = tk.Frame(sidebar, bg=self.COLORS['text_white'], height=1)
        sep2.pack(fill="x", pady=10)
        
        # Current Alert
        alert_title = tk.Label(sidebar, text="ALERTS", font=("Segoe UI", 11, "bold"),
                              bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        alert_title.pack(pady=(10, 10), padx=10)
        
        self.alert_label = tk.Label(sidebar, text="None", font=("Segoe UI", 9),
                                   bg=self.COLORS['bg_light'], fg=self.COLORS['text_gray'],
                                   wraplength=220, justify="left")
        self.alert_label.pack(pady=5, padx=10)
        
        # Separator line
        sep3 = tk.Frame(sidebar, bg=self.COLORS['text_white'], height=1)
        sep3.pack(fill="x", pady=10)
        
        # Hazard Indicator
        hazard_title = tk.Label(sidebar, text="HAZARD LEVEL", font=("Segoe UI", 11, "bold"),
                               bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        hazard_title.pack(pady=(10, 10), padx=10)
        
        self.hazard_indicator = tk.Label(sidebar, text="●", font=("Segoe UI", 28),
                                        bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        self.hazard_indicator.pack(pady=10)
        
        self.hazard_text = tk.Label(sidebar, text="SAFE", font=("Segoe UI", 10, "bold"),
                                   bg=self.COLORS['bg_light'], fg=self.COLORS['text_white'])
        self.hazard_text.pack(pady=5)
    
    def create_status_bar(self):
        """Create bottom status bar"""
        status_frame = tk.Frame(self.window, bg=self.COLORS['bg_light'], height=50)
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)
        
        # Top border
        border = tk.Frame(self.window, bg=self.COLORS['text_white'], height=2)
        border.pack(side="bottom", fill="x", before=status_frame)
        
        self.status_label = tk.Label(status_frame, text="System Ready - Click START to begin",
                                    font=("Segoe UI", 10), bg=self.COLORS['bg_light'],
                                    fg=self.COLORS['text_white'])
        self.status_label.pack(side="left", padx=15, pady=10)

    def get_direction(self, x1, x2, frame_width):
        """
        Determines if object is Left, Center, or Right.
        """
        center_x = (x1 + x2) / 2
        third = frame_width / 3
        
        if center_x < third:
            return "on your left"
        elif center_x > (third * 2):
            return "on your right"
        else:
            return "ahead"

    def get_audio_phrase(self, class_name, distance, direction, group):
        """
        Generates specific phrases based on the document tables.
        """
        # G1: Critical Hazards (Loud & Urgent)
        if group == 'G1':
            if class_name == 'fire':
                return f"Warning! Fire detected {distance} meters {direction}. Stop immediately."
            if class_name in ['gun', 'knife']:
                return f"Danger! Weapon detected {distance} meters {direction}. Move away."

        # G6: Traffic (Distance sensitive)
        elif group == 'G6':
            if distance < 3.0:
                return f"Stop! {class_name} approaching {direction}, {distance} meters."
            return f"{class_name} {distance} meters {direction}."

        # G3: Construction
        elif group == 'G3':
            return f"Caution. Construction hazard {distance} meters {direction}."

        # G2 & G4: Navigation / Indoor
        elif group in ['G2', 'G4']:
            return f"{class_name} {distance} meters {direction}."

        # Default fallback
        return f"{class_name} {distance} meters {direction}."
    
    def start_system(self):
        self.is_running = True
        self.btn_start.config(state="disabled", bg="#cccccc")
        self.btn_stop.config(state="normal")
        self.status_label.config(text="System Active - Scanning for Hazards...", fg=self.COLORS['text_white'])
        self.hazard_text.config(text="SCANNING", fg=self.COLORS['text_white'])
        self.hazard_indicator.config(fg=self.COLORS['text_white'])

    def stop_system(self):
        self.is_running = False
        self.btn_start.config(state="normal", bg=self.COLORS['text_white'])
        self.btn_stop.config(state="disabled", bg="#cccccc")
        self.status_label.config(text="System Paused", fg=self.COLORS['text_gray'])
        self.hazard_text.config(text="PAUSED", fg=self.COLORS['text_gray'])
        self.hazard_indicator.config(fg=self.COLORS['text_gray'])

    def toggle_audio(self):
        self.audio_enabled = not self.audio_enabled
        txt = "AUDIO" if self.audio_enabled else "MUTED"
        self.btn_audio.config(text=txt)
        status = "Audio enabled" if self.audio_enabled else "Audio muted"
        self.status_label.config(text=status, fg=self.COLORS['text_white'])

    def speech_worker(self):
        """
        Runs in the background. 
        If urgent -> Plays BEEP -> Then speaks.
        If normal -> Just speaks.
        """
        while True:
            # Wait for the next message
            item = self.speech_queue.get()
            
            if item is None: 
                break
            
            # Unpack the data: (The text to say, Is it urgent?)
            text, is_urgent = item
            
            try:
                # 1. PLAY BEEP (Only if urgent)
                if is_urgent and self.alert_sound:
                    self.alert_sound.play()
                    time.sleep(0.5) # Wait 0.5s so the beep finishes before voice starts
                
                # 2. SPEAK TEXT
                self.engine.say(text)
                self.engine.runAndWait()
                
            except RuntimeError:
                pass # Ignore errors if engine is busy
            except Exception as e:
                print(f"Audio Error: {e}")
            
            self.speech_queue.task_done()

    def speak_warning(self, text, priority_level):
        """
        Sends message to the worker. 
        Priority 4 (Fire/Knife) = URGENT (Beep + Speak Immediately).
        """
        current_time = time.time()
        
        # Level 4 is Critical (Defined in your config/logic)
        is_urgent = (priority_level >= 4)
        
        if self.audio_enabled:
            # --- CASE 1: CRITICAL HAZARD (FIRE/KNIFE) ---
            if is_urgent:
                try:
                    self.engine.stop() # Stop current sentence
                    with self.speech_queue.mutex:
                        self.speech_queue.queue.clear() # Delete less important messages
                except:
                    pass
                
                # Send (Text, True) -> True tells worker to BEEP
                self.speech_queue.put((text, True)) 
                self.last_speech_time = current_time

            # --- CASE 2: NORMAL OBJECT (CHAIR/TABLE) ---
            # Only speak if cooldown has passed
            elif current_time - self.last_speech_time > self.speech_cooldown:
                # Send (Text, False) -> False means NO beep
                self.speech_queue.put((text, False))
                self.last_speech_time = current_time

    def get_risk_action(self, group, distance):
        # Using logic from your config file structure
        severity = config.RISK_LEVELS.get(group, "LOW")
        
        if distance < 1.0 and severity in ["VERY_HIGH", "HIGH"]:
            return "STOP", (0, 0, 255) # Red
        elif distance < 3.0 and severity in ["VERY_HIGH", "HIGH", "MEDIUM"]:
            return "WARN", (0, 165, 255) # Orange
        return "INFO", (0, 255, 0) # Green

    def update_video(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame_height, frame_width, _ = frame.shape
                
                # FPS Calculation
                self.frame_count += 1
                if time.time() - self.fps_time > 1:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_time = time.time()
                
                # 1. Detect Objects
                results = self.model(frame, stream=True, conf=0.4, verbose=False)
                
                highest_priority = 0 
                audio_message = ""
                detections_count = 0

                for r in results:
                    for box in r.boxes:
                        detections_count += 1
                        # --- EXTRACT DATA ---
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        
                        # --- DISTANCE CALCULATION ---
                        real_h = config.OBJECT_HEIGHTS.get(class_name, 1.0)
                        bbox_h = y2 - y1
                        dist = round((real_h * self.focal_length) / bbox_h, 1) if bbox_h > 0 else 0

                        # --- CONTEXT ---
                        group = config.CLASS_TO_GROUP.get(class_name, 'G8')
                        direction = self.get_direction(x1, x2, frame_width)

                        # --- RISK CALCULATION ---
                        risk_level = 1 
                        
                        if group in ['G1']: risk_level = 4
                        elif group in ['G6'] and dist < 4.0: risk_level = 4
                        elif group in ['G3'] and dist < 2.0: risk_level = 3
                        elif dist < 1.0: risk_level = 3
                        elif group in ['G4', 'G7']: risk_level = 2

                        # --- VISUALIZATION ---
                        if risk_level == 4: color = (0, 0, 255)
                        elif risk_level == 3: color = (0, 165, 255)
                        else: color = (0, 255, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_name} {dist}m", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # --- AUDIO DECISION ---
                        if risk_level > highest_priority:
                            highest_priority = risk_level
                            audio_message = self.get_audio_phrase(class_name, dist, direction, group)

                # 2. Trigger Audio
                if highest_priority > 0 and audio_message:
                    self.speak_warning(audio_message, highest_priority)

                # Update sidebar
                self.fps_label.config(text=f"FPS: {self.fps}")
                self.obj_label.config(text=f"Objects: {detections_count}")
                
                # Update hazard level indicator
                if highest_priority == 4:
                    self.hazard_indicator.config(fg="#ffffff")
                    self.hazard_text.config(text="CRITICAL", fg="#ffffff")
                    self.alert_label.config(text=audio_message, fg="#ffffff")
                elif highest_priority == 3:
                    self.hazard_indicator.config(fg="#dddddd")
                    self.hazard_text.config(text="WARNING", fg="#dddddd")
                    self.alert_label.config(text=audio_message, fg="#dddddd")
                elif highest_priority == 2:
                    self.hazard_indicator.config(fg="#cccccc")
                    self.hazard_text.config(text="INFO", fg="#cccccc")
                    self.alert_label.config(text=audio_message, fg="#cccccc")
                else:
                    self.hazard_indicator.config(fg="#aaaaaa")
                    self.hazard_text.config(text="SAFE", fg="#aaaaaa")
                    self.alert_label.config(text="No hazards detected", fg=self.COLORS['text_gray'])

                # 4. Display on GUI
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)

        self.window.after(30, self.update_video)

    def on_close(self):
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = NavAssistApp(root, "NavAssist AI - Safety System")
    root.mainloop()