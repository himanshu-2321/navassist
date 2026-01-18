# NavAssist AI: Intelligent Blind Navigation System 👁️🤖

**NavAssist AI** is a real-time computer vision system designed to aid visually impaired individuals. It uses **YOLOv8** for object detection and a custom risk assessment algorithm to identify hazards, estimate their distance, and provide prioritized audio feedback (Text-to-Speech + Alert Beeps).

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-purple)

## 🚀 Features

* **Real-Time Object Detection:** Identifies pedestrians, vehicles, furniture, and critical hazards (Fire, Weapons) using YOLOv8.
* **Distance & Direction Estimation:** Calculates approximate distance in meters and determines if an object is to the **Left**, **Right**, or **Ahead**.
* **Dynamic Risk Assessment:**
    * **Critical Hazards (Fire, Fast Cars):** Triggers immediate "STOP" alerts with a loud beep.
    * **Navigation Obstacles (Chairs, Stairs):** Provides gentle verbal warnings.
    * [cite_start]**Logic:** Risk is calculated as `Hazard Severity × Distance`[cite: 37].
* [cite_start]**Priority Audio Engine:** Critical warnings override standard navigation instructions to ensure immediate reaction to danger[cite: 41].
* **High-Contrast Dashboard:** A dark-mode GUI ("Command Center" style) designed for clarity and low-light environments.

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* A working webcam

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/NavAssist-AI.git](https://github.com/yourusername/NavAssist-AI.git)
cd NavAssist-AI

```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python pyttsx3 pygame pillow

```

### 3. Add Model & Audio Files

Ensure the following files are in your project root folder:

* `best.pt` (Your trained YOLO model) OR `yolov8n.pt` (Standard model)
* `beep-beep-6151.mp3` (Alert sound file)

## 💻 Usage

Run the main application:

```bash
python gui_app.py

```
### Alternative Usage (Lightweight Mode)
If you want to run the system without the full Dashboard UI (e.g., for testing or slower PCs):
```bash
python main.py

### Controls

* **START:** Initializes the camera and AI engine.
* **STOP:** Pauses the video feed and processing.
* **AUDIO/MUTE:** Toggles voice feedback.
* **ESC / Close Window:** Exits the application safely.

## 🧠 How It Works

The system follows a strict decision-making flowchart:

1. **Capture:** Webcam feeds video to the YOLOv8 model.
2. **Analyze:** The model detects objects and draws bounding boxes.
3. 
**Distance Calculation:** Using focal length estimation logic: `Distance = (Real_Height * Focal_Length) / Image_Height`.


4. 
**Risk Grouping:** Objects are categorized into **8 Safety Groups** (e.g., G1=Critical, G6=Traffic).


5. **Audio Output:**
* **Level 4 (Critical):** Plays `beep.mp3` -> "Stop immediately!"
* **Level 2-3 (Warning):** Speaks "Chair ahead, 2 meters."



## 📂 Project Structure

```
NavAssist-AI/
│
├── gui_app.py          # Main application (UI, Camera, Audio Logic)
├── config.py           # Configuration (Risk levels, Object heights, Groups)
├── beep-beep-6151.mp3  # Emergency alert sound
├── best.pt             # YOLOv8 Weights file
└── README.md           # Documentation

```

## ⚙️ Configuration

You can customize safety settings in `config.py`:

* 
**`OBJECT_HEIGHTS`**: Adjust real-world heights (in meters) for better distance accuracy.


* **`RISK_LEVELS`**: Change which groups trigger "STOP" vs "WARN".
* **`CLASS_TO_GROUP`**: Map new YOLO classes to specific safety groups (e.g., adding 'scooter' to Group 6).

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

* **Ultralytics** for the YOLOv8 model.
* **Pygame & pyttsx3** for the audio capabilities.
* Safety logic based on the "NavAssist Hazard Protocol" documentation.

```

```
