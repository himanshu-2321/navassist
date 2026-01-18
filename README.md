# NavAssist — Intelligent Navigation Assistant for the Visually Impaired 👁️🤖

NavAssist is a real-time computer vision system that helps visually impaired users navigate safely. It uses YOLOv8 for object detection plus a configurable risk-assessment engine to provide spoken guidance, directional cues, and urgent audio alerts for critical hazards.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-purple)

---

## Key Features

- Real-time object detection (pedestrians, vehicles, furniture, hazards) via YOLOv8.
- Approximate distance estimation and direction (Left / Right / Ahead).
- Dynamic risk assessment with configurable severity groups:
  - Critical hazards (e.g., fire, fast-moving vehicles) trigger immediate "STOP" alerts and loud beeps.
  - Navigation obstacles (e.g., chairs, stairs) produce gentle verbal warnings.
- Priority audio engine — critical alerts override regular navigation instructions.
- High-contrast, dark-mode GUI "Command Center" for low-light readability.
- Lightweight CLI mode for low-power or headless operation.

---

## Quickstart

### Prerequisites

- Python 3.8 or newer
- A working webcam (or video source)
- Optional: a virtual environment (recommended)

### 1. Clone the repository

```bash
git clone https://github.com/himanshu-2321/navassist.git
cd navassist
```

### 2. Install dependencies

If a `requirements.txt` exists:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or install core packages directly:

```bash
pip install ultralytics opencv-python pyttsx3 pygame pillow
```

> Note: Depending on your platform, additional system packages (audio drivers, ffmpeg) may be required.

### 3. Add model & audio files

Place these files in the repository root (or update paths in `config.py`):

- `best.pt` — your trained YOLOv8 weights (or)
- `yolov8n.pt` — standard YOLOv8 weight (small)
- `beep-beep-6151.mp3` — emergency alert sound (or update the filename in config)

---

## Usage

Run the GUI application (recommended):

```bash
python gui_app.py
```

Lightweight (no GUI):

```bash
python main.py
```

Controls (GUI)
- START: initialize camera and AI engine
- STOP: pause feed & processing
- AUDIO / MUTE: toggle voice feedback
- ESC or Close Window: exit safely

---

## How it works (overview)

1. Capture: Webcam streams frames to YOLOv8.
2. Detect: YOLOv8 returns detections with class and bounding box.
3. Distance estimation: focal-length based calculation is used to estimate distance (meters).

   Distance formula:
   ```
   Distance = (Real_Height * Focal_Length) / Image_Height
   ```

   - Real_Height: configured real-world object height (m)
   - Focal_Length: camera-specific focal length (calibrated)
   - Image_Height: height of object's bounding box in pixels

4. Risk assessment:
   - Objects are mapped to safety groups (e.g., G1 = Critical).
   - A risk score is computed; higher scores produce more urgent alerts.

   Example risk logic:
   ```
   Risk = Hazard_Severity × (1 / Distance)
   ```
   (Configurable in `config.py` — you can change severity weights and thresholds.)

5. Audio & UI:
   - Critical (Level 4): plays beep + immediate spoken "Stop immediately!"
   - Warning (Level 2–3): spoken guidance, e.g., "Chair ahead, 2 meters."
   - Navigation instructions are suppressed while critical alerts are active.

---

## Configuration

Edit `config.py` to customize behavior:

- OBJECT_HEIGHTS: dictionary of real-world object heights (meters) used for distance calculations.
- RISK_LEVELS: thresholds that determine STOP vs WARN vs IGNORE actions.
- CLASS_TO_GROUP: map YOLO class names to safety groups (add new classes as needed).
- AUDIO_SETTINGS: volume, voice rate, and priority overrides.

Make small changes and test in lightweight mode before using the GUI.

---

## Project structure

```
navassist/
│
├── gui_app.py          # Main GUI application (UI, camera loop, audio & alert logic)
├── main.py             # Lightweight headless runner for testing / low-power env
├── config.py           # Configurable constants: groups, heights, thresholds
├── beep-beep-6151.mp3  # Emergency alert sound (example file)
├── best.pt             # YOLOv8 weights (user-supplied/trained)
├── requirements.txt    # (optional) Python dependencies
└── README.md           # This documentation
```

(Adjust names if your repo structure differs.)

---

## Development & Contribution

Contributions, issues, and feature requests are welcome.

Suggested workflow:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add: short description"`
4. Push branch: `git push origin feature/your-feature`
5. Open a Pull Request describing the change and its motivation

Please follow Python best practices (virtualenv, linting, small commits) and include tests where applicable.

---

## Testing & Safety

- Test with a calibrated camera for best distance estimates.
- Verify `OBJECT_HEIGHTS` values for key classes (person, chair, stair).
- Use caution and always validate system behavior in a controlled environment before relying on it for real-world navigation.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- Ultralytics — for YOLOv8
- pyttsx3 & pygame — audio & playback
- Inspiration: "NavAssist Hazard Protocol"

---

If you'd like, I can:
- produce a cleaned `requirements.txt` from the imports in your codebase,
- suggest improved defaults for `config.py` (object heights and risk thresholds),
- or rewrite the GUI help overlay text for clarity. Which would you prefer next?
