# RoadDamage AI Dashboard

A Flask web application for road damage detection using YOLOv8.

## What it does

- Upload road images (drag & drop or file browser)
- Run YOLOv8 detection to identify 5 types of road damage
- View annotated results with confidence scores
- Compare YOLOv8, YOLOv7, and TPH-YOLOv5 performance metrics

## Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the app:
```bash
python app.py
```

3. Open `http://127.0.0.1:5000`

## Model

The dashboard loads the YOLOv8 checkpoint from `3_Trained_Models_and_Logs/yolov8/best.pt`.
