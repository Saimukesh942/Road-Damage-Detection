Road Damage Detection
This folder contains the RoadDamage AI project: a web dashboard for detecting road damage in images, a model-training pipeline, and trained model artifacts.

Project contents
RoadDamage_AI_Release/ — the project release bundle.
1_Dashboard_Application/ — Flask dashboard, templates, static assets, and Python dependencies.
2_Model_Training_Pipeline/ — Kaggle notebook and exported Python training script.
3_Trained_Models_and_Logs/ — trained model checkpoints and experiment logs.
README.txt — release bundle overview.
Run the dashboard
Windows
Ensure Python and pip are installed and available on your PATH.
Run RUN_DASHBOARD.bat.
Open http://127.0.0.1:5000 in your browser.
The batch file installs the dashboard dependencies and starts the Flask server. To install them manually, open a terminal in RoadDamage_AI_Release\1_Dashboard_Application and run:

python -m pip install -r requirements.txt
python app.py
The dashboard accepts road images for inference and displays annotated detections and model comparison information. The default inference checkpoint is the final five-class YOLOv8 model in 3_Trained_Models_and_Logs\yolov8\5-class-split-50-epoch\best_v2_final.pt.

Model training
The notebook in 2_Model_Training_Pipeline documents the dataset preparation, training, and evaluation workflow for the evaluated YOLO architectures. It is designed for Kaggle; the notebook notes that internet access is needed for first-run downloads and that the RDD2022 dataset must be attached as a Kaggle input.
