@echo off
title RoadDamage AI Dashboard Start Script
echo =======================================================
echo          ROADDAMAGE AI DASHBOARD LAUNCHER
echo =======================================================
echo.
echo Installing required Python packages...
cd 1_Dashboard_Application
pip install -r requirements.txt
echo.
echo Starting the Flask Server...
echo The dashboard will be available at http://127.0.0.1:5000/
echo =======================================================
python app.py
pause
