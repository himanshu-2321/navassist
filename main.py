import cv2
import pyttsx3
import time
from ultralytics import YOLO
import config

# --- SETUP ---
# Initialize Audio Engine
engine = pyttsx3.init()

# Load Model (Downloads a small AI model the first time)
print("Loading AI Model...")
model = YOLO('yolov8n.pt') 

# Start Camera
cap = cv2.VideoCapture(0)
focal_length = 600  # Standard webcam focal length

def get_risk_action(group, distance):
    """
    Decides if we should STOP, WARN, or just INFORM
    [cite_start]Based on Logic Matrix [cite: 38]
    """
    severity = config.RISK_LEVELS.get(group, "LOW")
    
    # [cite_start]Distance Logic [cite: 36, 38]
    if distance < 1.0: # D1 Zone (Immediate)
        if severity == "VERY_HIGH": return "STOP", (0, 0, 255) # Red
        if severity == "HIGH": return "STOP", (0, 0, 255)
    
    if distance < 3.0: # D2 Zone (Near)
        if severity in ["VERY_HIGH", "HIGH"]: return "WARN", (0, 120, 255) # Orange
        if severity == "MEDIUM": return "WARN", (0, 255, 255) # Yellow
        
    return "INFO", (0, 255, 0) # Green

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera not found.")
        break

    # Detect Objects
    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # 1. Get Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 2. Identify Object
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # [cite_start]3. Calculate Distance [cite: 40]
            bbox_height = y2 - y1
            real_height = config.OBJECT_HEIGHTS.get(class_name, 1.0)
            
            if bbox_height > 0:
                dist = round((real_height * focal_length) / bbox_height, 1)
            else:
                dist = 0

            # 4. Determine Risk
            group = config.CLASS_TO_GROUP.get(class_name, "G2") # Default to Navigation group
            action, color = get_risk_action(group, dist)

            # 5. Draw on Screen
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.upper()} {dist}m"
            
            # Add 'STOP' text if dangerous
            if action == "STOP":
                label += " [STOP!]"
                # Simple Audio Alert (Prints to console to avoid noise spam)
                print(f"⚠️ DANGER: {class_name} is {dist}m away!") 
                
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("NavAssist AI", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()