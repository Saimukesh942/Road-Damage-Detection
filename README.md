# Road Damage Detection

This folder contains the RoadDamage AI project: a web dashboard for detecting road damage in images, a model-training pipeline, and trained model artifacts.

## Project contents

- [`RoadDamage_AI_Release/`](./RoadDamage_AI_Release/) — the project release bundle.
  - [`1_Dashboard_Application/`](./RoadDamage_AI_Release/1_Dashboard_Application/) — Flask dashboard, templates, static assets, and Python dependencies.
  - [`2_Model_Training_Pipeline/`](./RoadDamage_AI_Release/2_Model_Training_Pipeline/) — Kaggle notebook and exported Python training script.
  - [`3_Trained_Models_and_Logs/`](./RoadDamage_AI_Release/3_Trained_Models_and_Logs/) — trained model checkpoints and experiment logs.
  - [`README.txt`](./RoadDamage_AI_Release/README.txt) — release bundle overview.

## Run the dashboard

### Windows

1. Ensure Python and `pip` are installed and available on your `PATH`.
2. Run [`RUN_DASHBOARD.bat`](./RoadDamage_AI_Release/RUN_DASHBOARD.bat).
3. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

The batch file installs the dashboard dependencies and starts the Flask server. To install them manually, open a terminal in `RoadDamage_AI_Release\1_Dashboard_Application` and run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

The dashboard accepts road images for inference and displays annotated detections and model comparison information. The default inference checkpoint is the final five-class YOLOv8 model in `3_Trained_Models_and_Logs\yolov8\5-class-split-50-epoch\best_v2_final.pt`.

## Model training

The notebook in `2_Model_Training_Pipeline` documents the dataset preparation, training, and evaluation workflow for the evaluated YOLO architectures. It is designed for Kaggle; the notebook notes that internet access is needed for first-run downloads and that the RDD2022 dataset must be attached as a Kaggle input.
