from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
DATABASE_PATH = ROOT / "road_damage_auth.sqlite3"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RUNTIME_DIR = ROOT / ".runtime"
MODEL_REGISTRY = {
    "yolov8_final_5class": {
        "label": "YOLOv8",
        "path": WORKSPACE_ROOT / "3_Trained_Models_and_Logs" / "yolov8" / "5-class-split-50-epoch" / "best_v2_final.pt",
    },
    "yolov8_small_6class": {
        "label": "YOLOv8 Exploratory 6-Class",
        "path": WORKSPACE_ROOT / "3_Trained_Models_and_Logs" / "yolov8" / "best.pt",
    },
}
DEFAULT_MODEL_ID = "yolov8_final_5class"

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SECRET_KEY"] = os.environ.get("ROAD_DAMAGE_SECRET_KEY", "road-damage-dev-secret")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

try:
    from ultralytics import YOLO

    ULTRALYTICS_IMPORT_ERROR = ""
except Exception as exc: 
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = f"{exc.__class__.__name__}: {exc}"

_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


def init_db() -> None:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                thumbnail TEXT NOT NULL,
                annotated_thumbnail TEXT NOT NULL,
                damage_count INTEGER NOT NULL DEFAULT 0,
                top_confidence REAL NOT NULL DEFAULT 0,
                detections_json TEXT NOT NULL DEFAULT '[]',
                searched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@app.teardown_appcontext
def close_db(_exception) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


@app.before_request
def load_current_user() -> None:
    user_id = session.get("user_id")
    g.current_user = get_user_by_id(user_id) if user_id else None


def get_user_by_email(email: str) -> sqlite3.Row | None:
    return get_db().execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()


def get_user_by_id(user_id: int | None) -> sqlite3.Row | None:
    if not user_id:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def create_user(full_name: str, email: str, password: str) -> int:
    cursor = get_db().execute(
        "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
        (full_name.strip(), email.lower().strip(), generate_password_hash(password)),
    )
    get_db().commit()
    return int(cursor.lastrowid)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.current_user is None:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def api_auth_required() -> tuple[bool, Any]:
    if g.current_user is None:
        return False, (jsonify({"ok": False, "error": "Login required."}), 401)
    return True, None


def model_status(model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
    model_meta = MODEL_REGISTRY.get(model_id)
    if model_meta is None:
        return {"ready": False, "message": "Unknown model id.", "details": model_id}
    if YOLO is None:
        return {
            "ready": False,
            "message": "Ultralytics is not installed.",
            "details": ULTRALYTICS_IMPORT_ERROR,
        }
    if not model_meta["path"].exists():
        return {
            "ready": False,
            "message": "Model weights were not found.",
            "details": str(model_meta["path"]),
        }
    return {
        "ready": True,
        "message": "Detection engine is ready.",
        "details": str(model_meta["path"]),
    }


def all_model_statuses() -> list[dict[str, Any]]:
    rows = []
    for model_id, meta in MODEL_REGISTRY.items():
        rows.append({"id": model_id, "label": meta["label"], **model_status(model_id)})
    return rows


def ensure_model(model_id: str = DEFAULT_MODEL_ID):
    status = model_status(model_id)
    if not status["ready"]:
        raise RuntimeError(f"{status['message']} {status['details']}".strip())
    if model_id in _model_cache:
        return _model_cache[model_id]
    with _model_lock:
        if model_id not in _model_cache:
            _model_cache[model_id] = YOLO(str(MODEL_REGISTRY[model_id]["path"]))
    return _model_cache[model_id]


def allowed_image(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in ALLOWED_SUFFIXES


def decode_image(raw_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the uploaded image.")
    return image


def encode_png_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Unable to encode the annotated image.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def label_for_class(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def class_guide() -> list[dict[str, str]]:
    return [
        {
            "code": "D00",
            "title": "Longitudinal crack",
            "description": "Linear cracking that runs along the road direction.",
        },
        {
            "code": "D10",
            "title": "Transverse crack",
            "description": "Cracks crossing the lane width from side to side.",
        },
        {
            "code": "D20",
            "title": "Alligator crack",
            "description": "Dense interconnected cracking that resembles a web pattern.",
        },
        {
            "code": "D40",
            "title": "Pothole or severe surface defect",
            "description": "Pothole-like or heavily degraded surface regions.",
        },
        {
            "code": "Repair",
            "title": "Patched repair area",
            "description": "Previously repaired or patched pavement sections.",
        },
    ]


@app.get("/")
def home():
    return render_template(
        "marketing.html",
        model_status=model_status(DEFAULT_MODEL_ID),
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.current_user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not full_name:
            flash("Enter your full name.", "error")
        elif not email or "@" not in email:
            flash("Enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif get_user_by_email(email) is not None:
            flash("That email is already registered.", "error")
        else:
            user_id = create_user(full_name, email, password)
            session.clear()
            session["user_id"] = user_id
            flash("Account created successfully.", "success")
            return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = get_user_by_email(email)

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.", "error")
        else:
            session.clear()
            session["user_id"] = int(user["id"])
            flash(f"Welcome back, {user['full_name'].split()[0]}.", "success")
            return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.post("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.get("/dashboard")
@login_required
def dashboard():
    quick_stats = [
        {"label": "Model", "value": "YOLOv8", "note": "5-class detection model"},
        {"label": "Coverage", "value": "5 classes", "note": "D00, D10, D20, D40, Repair"},
        {"label": "Input flow", "value": "Drag + browse", "note": "Upload, replace, or clear instantly"},
    ]
    return render_template(
        "dashboard.html",
        user=g.current_user,
        user_first_name=g.current_user["full_name"].split()[0],
        model_label=MODEL_REGISTRY[DEFAULT_MODEL_ID]["label"],
        model_status=model_status(DEFAULT_MODEL_ID),
        quick_stats=quick_stats,
        class_guide=class_guide(),
    )


@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "app": "road-damage-saas-dashboard",
            "defaultModel": model_status(DEFAULT_MODEL_ID),
            "models": all_model_statuses(),
            "authenticated": g.current_user is not None,
        }
    )


def make_thumbnail_data_url(image: np.ndarray, max_size: int = 160) -> str:
    """Resize an image to a small thumbnail and return as a data URL."""
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    thumb = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


@app.post("/api/predict")
def api_predict():
    allowed, response = api_auth_required()
    if not allowed:
        return response

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "Upload an image first."}), 400
    if not allowed_image(uploaded.filename):
        return jsonify({"ok": False, "error": "Use JPG, PNG, BMP, or WEBP."}), 400

    raw_bytes = uploaded.read()
    if not raw_bytes:
        return jsonify({"ok": False, "error": "The uploaded file is empty."}), 400

    try:
        image = decode_image(raw_bytes)
        model = ensure_model(DEFAULT_MODEL_ID)
        result = model.predict(source=image, imgsz=960, conf=0.25, verbose=False, save=False)[0]
        annotated = result.plot()

        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = [int(class_id) for class_id in boxes.cls.cpu().tolist()]
            for box, score, class_id in zip(xyxy, confs, classes):
                detections.append(
                    {
                        "label": label_for_class(result.names, class_id),
                        "confidence": round(float(score), 4),
                        "box": [round(float(value), 2) for value in box],
                    }
                )

        # Save to search history
        try:
            orig_thumb = make_thumbnail_data_url(image)
            ann_thumb = make_thumbnail_data_url(annotated)
            sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
            top_conf = sorted_dets[0]["confidence"] if sorted_dets else 0
            get_db().execute(
                "INSERT INTO search_history (user_id, filename, thumbnail, annotated_thumbnail, damage_count, top_confidence, detections_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session.get("user_id"),
                    uploaded.filename,
                    orig_thumb,
                    ann_thumb,
                    len(detections),
                    top_conf,
                    json.dumps(detections),
                ),
            )
            get_db().commit()
        except Exception:
            pass  

        return jsonify(
            {
                "ok": True,
                "modelLabel": MODEL_REGISTRY[DEFAULT_MODEL_ID]["label"],
                "classNames": [label_for_class(result.names, idx) for idx in sorted(result.names.keys())]
                if isinstance(result.names, dict)
                else [str(name) for name in result.names],
                "annotatedImage": encode_png_data_url(annotated),
                "imageWidth": int(image.shape[1]),
                "imageHeight": int(image.shape[0]),
                "count": len(detections),
                "detections": detections,
            }
        )
    except Exception as exc:  
        return jsonify({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}), 500


@app.get("/api/history")
def api_history():
    allowed, response = api_auth_required()
    if not allowed:
        return response

    rows = get_db().execute(
        "SELECT id, filename, thumbnail, annotated_thumbnail, damage_count, top_confidence, detections_json, searched_at FROM search_history WHERE user_id = ? ORDER BY searched_at DESC LIMIT 20",
        (session.get("user_id"),),
    ).fetchall()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "filename": row["filename"],
            "thumbnail": row["thumbnail"],
            "annotatedThumbnail": row["annotated_thumbnail"],
            "damageCount": row["damage_count"],
            "topConfidence": round(row["top_confidence"] * 100),
            "detections": json.loads(row["detections_json"]),
            "searchedAt": row["searched_at"],
        })

    return jsonify({"ok": True, "history": history})


@app.post("/api/history/delete/<int:entry_id>")
def api_delete_history_entry(entry_id: int):
    allowed, response = api_auth_required()
    if not allowed:
        return response

    get_db().execute(
        "DELETE FROM search_history WHERE id = ? AND user_id = ?",
        (entry_id, session.get("user_id")),
    )
    get_db().commit()
    return jsonify({"ok": True})


@app.post("/api/history/clear")
def api_clear_history():
    allowed, response = api_auth_required()
    if not allowed:
        return response

    get_db().execute("DELETE FROM search_history WHERE user_id = ?", (session.get("user_id"),))
    get_db().commit()
    return jsonify({"ok": True})


init_db()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
