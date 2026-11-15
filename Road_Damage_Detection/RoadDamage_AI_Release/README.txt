# RoadDamage AI

Welcome to the RoadDamage AI project. This directory contains the complete artifact bundle of the dashboard, our reproduction pipelines, trained architectures, and empirical validations.

## 📂 Directory Structure

**`1_Dashboard_Application`**
The core Flask/SQLite web application. This acts as the final inference workspace and SaaS marketing page.
- Run `RUN_DASHBOARD.bat` in the root folder to auto-install dependencies and start the app on port 5000.

**`2_Model_Training_Pipeline`**
Contains the exhaustive Kaggle reproduction notebook and the exported Python script (`road_damage_kaggle_notebook`). This is the singular pipeline used to train all evaluated architectures (YOLOv8, YOLOv7, TPH-YOLOv5) on the identical `merged_uav_paper_style` dataset.

**`3_Trained_Models_and_Logs`**
Contains the raw training outputs, weight files (`best.pt`), and evaluation logs directly mapped from the Kaggle environment to back our dashboard's metrics.
- YOLOv8 (inside `yolov8`)
- YOLOv7 (inside `yolov7`)
- TPH-YOLOv5 (inside `final_model_yolov5_tph`)

**`4_Research_Paper`**
The base IEEE Access 2023 paper by Silva et al., provided as the foundational benchmark context.

## 🚀 How to Run
1. Double-click the **`RUN_DASHBOARD.bat`** file.
2. It will automatically initialize the environment, download prerequisites via `pip`, and spin up the server.
3. Open your browser and navigate to `http://localhost:5000` to interact with the application.
