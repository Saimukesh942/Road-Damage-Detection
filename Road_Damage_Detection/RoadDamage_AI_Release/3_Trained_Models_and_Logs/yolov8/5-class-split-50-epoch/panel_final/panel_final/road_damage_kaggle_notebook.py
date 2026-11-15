# %% [markdown]
# # Automated Road Damage Detection From UAV Images
#
# This notebook reproduces the baseline training setup from:
# - Silva et al., "Automated Road Damage Detection Using UAV Images and Deep Learning Techniques" (IEEE Access, 2023): https://gredos.usal.es/bitstream/handle/10366/161506/Automated_Road_Damage_Detection_Using_UAV_Images_and_Deep_Learning_Techniques.pdf?isAllowed=y&sequence=1
# - Spain UAV source repository used by the paper: https://github.com/luisaugustos/Pothole-Recognition
# - YOLOv5 v7.0: https://github.com/ultralytics/yolov5/tree/v7.0
# - Official TPH-YOLOv5: https://github.com/cv516Buaa/tph-yolov5
# - Official YOLOv7: https://github.com/WongKinYiu/yolov7
# - Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
#
# It reconstructs the Spain UAV dataset locally from the original repository, converts the China UAV subset of RDD2022 into a clean YOLO working set, merges both sources, builds paper-faithful train/val/test splits, runs model-specific training, and evaluates all models with a common post-hoc evaluator.
#
# Notes:
# - Kaggle Internet should be ON for the first run because the notebook clones model repositories and downloads pretrained weights.
# - RDD2022 should be attached as a Kaggle input dataset.
# - The paper has a few count inconsistencies. This notebook logs raw image counts, augmented image counts, label counts, split counts, and any mismatch instead of hiding them.

# %%
import importlib
import os
import random
import subprocess
import sys
from pathlib import Path

EARLY_IS_KAGGLE = Path("/kaggle").exists()


def pip_install_if_needed(package_specs):
    missing = []
    for module_name, package_spec in package_specs:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(package_spec)
    if missing:
        print("Installing missing packages:", missing)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)


if EARLY_IS_KAGGLE:
    BASE_PACKAGES = [
        ("yaml", "PyYAML>=6,<7"),
        ("albumentations", "albumentations>=1.4.8,<2.1"),
        ("torchmetrics", "torchmetrics>=1.4.1,<1.7"),
        ("ultralytics", "ultralytics>=8.3.0,<9"),
    ]
else:
    BASE_PACKAGES = [
        ("yaml", "PyYAML>=6,<7"),
        ("pandas", "pandas>=2,<3"),
        ("numpy", "numpy>=1.26"),
        ("cv2", "opencv-python-headless>=4.8"),
        ("albumentations", "albumentations>=1.4.8,<2.1"),
        ("PIL", "Pillow>=10,<12"),
        ("matplotlib", "matplotlib>=3.8,<4"),
        ("seaborn", "seaborn>=0.13,<0.14"),
        ("sklearn", "scikit-learn>=1.4,<1.6"),
        ("tqdm", "tqdm>=4.66,<5"),
        ("torchmetrics", "torchmetrics>=1.4.1,<1.7"),
        ("ultralytics", "ultralytics>=8.3.0,<9"),
    ]

pip_install_if_needed(BASE_PACKAGES)

# %%
import contextlib
import gc
import hashlib
import json
import math
import shutil
import socket
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("ULTRALYTICS_WANDB", "False")
sns.set_theme(style="whitegrid")

with contextlib.suppress(Exception):
    import ray.train._internal.session as ray_session

    if not hasattr(ray_session, "_get_session") and hasattr(ray_session, "get_session"):
        ray_session._get_session = ray_session.get_session

# %% [markdown]
# ## 1. Configuration
#
# `MODEL_NAME` selects the model family to train.
# Available values:
# - `yolov4`
# - `yolov5`
# - `yolov5_tph`
# - `yolov7`
# - `yolov8`
#
# Set `TRAIN_MODEL_SEQUENCE = True` and fill `MODEL_SEQUENCE` if you want to train more than one model serially in the same notebook run.

# %%
IS_KAGGLE = Path("/kaggle").exists()
KAGGLE_INPUT = Path("/kaggle/input") if IS_KAGGLE else Path.cwd() / "input"
KAGGLE_WORKING = Path("/kaggle/working") if IS_KAGGLE else Path.cwd() / "working"
NON_KAGGLE_ROOT_OUT = Path(os.environ["ROAD_DAMAGE_ROOT"]).expanduser() if os.environ.get("ROAD_DAMAGE_ROOT") else (
    Path("/home/road_damage/working") if Path("/home").exists() else Path.cwd() / "working"
)
ROOT_OUT_PATH = KAGGLE_WORKING / "road_damage_repro" if IS_KAGGLE else NON_KAGGLE_ROOT_OUT


@dataclass
class ModelSpec:
    model_name: str
    family: str
    display_name: str
    repo_name: Optional[str]
    repo_url: Optional[str]
    repo_ref: Optional[str]
    pretrained_weights: Optional[str]
    cfg: Optional[str] = None
    train_script: Optional[str] = None
    detect_script: Optional[str] = None
    hyp_file: Optional[str] = None
    batch_size: int = 16
    epochs: int = 300
    optimizer: str = "SGD"
    img_size: int = 640
    notes: str = ""


@dataclass
class Config:
    MODEL_NAME: str = "yolov8"
    TRAIN_MODEL_SEQUENCE: bool = False
    MODEL_SEQUENCE: List[str] = field(default_factory=lambda: ["yolov4", "yolov5", "yolov5_tph", "yolov7", "yolov8"])

    SEED: int = 42
    IMG_SIZE: int = 640
    EPOCHS_OVERRIDE: Optional[int] = None
    WORKERS: int = 8
    PATIENCE: int = 50
    SANITY_MODE: bool = False
    DRY_RUN: bool = False
    DRY_RUN_MAX_IMAGES_PER_SOURCE: int = 32
    DRY_RUN_EPOCHS: int = 1
    SAVE_DEBUG_PLOTS: bool = True
    CACHE_REPOS: bool = True
    REQUIRE_INTERNET_FOR_SETUP: bool = True
    FORCE_REBUILD_DATASETS: bool = False
    RESUME: bool = True

    TARGET_CLASSES: List[str] = field(default_factory=lambda: ["D00", "D10", "D20", "D40", "Repair"])
    MIN_SPLIT_IMAGES_PER_CLASS: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "train": {"D00": 1, "D10": 1, "D20": 1, "D40": 1, "Repair": 1},
        "val": {"D20": 5, "Repair": 3},
        "test": {"D10": 5, "D20": 5, "Repair": 3},
    })
    MIN_TRAIN_IMAGE_PRESENCE_AFTER_AUG: Dict[str, int] = field(default_factory=lambda: {
        "D20": 220,
        "Repair": 120,
    })
    MIN_TRAIN_CLASS_RATIO_AFTER_AUG: Dict[str, float] = field(default_factory=lambda: {
        "D20": 0.25,
        "Repair": 0.25,
    })
    YOLOV8_ACCURACY_MODE: bool = True
    YOLOV8_ACCURACY_IMGSZ: int = 960
    YOLOV8_EVAL_TTA: bool = False
    PAPER_REFERENCE_MAP50: Dict[str, float] = field(default_factory=lambda: {
        "yolov4": 0.268638,
        "yolov5": 0.5990,
        "yolov5_tph": 0.6570,
        "yolov7": 0.7370,
        "yolov8": float("nan"),
    })

    RDD2022_ROOT: Optional[str] = None
    SPAIN_REPO_URL: str = "https://github.com/luisaugustos/Pothole-Recognition.git"
    SPAIN_REPO_REF: str = "master"

    ROOT_OUT: str = str(ROOT_OUT_PATH)
    REPOS_DIR: str = str(ROOT_OUT_PATH / "repos")
    DATASETS_DIR: str = str(ROOT_OUT_PATH / "datasets")
    EXPERIMENTS_DIR: str = str(ROOT_OUT_PATH / "experiments")
    SUMMARIES_DIR: str = str(ROOT_OUT_PATH / "summaries")
    CONFIGS_DIR: str = str(ROOT_OUT_PATH / "configs")
    LOGS_DIR: str = str(ROOT_OUT_PATH / "logs")

    SPAIN_TARGET_RAW_IMAGES: int = 568
    SPAIN_TARGET_TOTAL_IMAGES: int = 1362
    SPAIN_ZOOM_RANGE: Tuple[float, float] = (0.00, 0.15)
    SPAIN_USE_ONLY_ZOOM_REBUILD: bool = True
    MERGED_ROTATION_RANGE: Tuple[float, float] = (-15.0, 15.0)
    TARGET_AUGMENTED_TRAIN_IMAGES: int = 4000
    SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.20, 0.10)

    CONFIDENCE_STANDARD: float = 0.25
    CONF_SWEEP_START: float = 0.05
    CONF_SWEEP_STOP: float = 0.95
    CONF_SWEEP_STEPS: int = 19
    IOU_EVAL: float = 0.50
    SPEED_EVAL_IMAGES: int = 64

    YOLOV8_PACKAGE_VERSION: str = "8.3.0"


CFG = Config()
CFG.MODEL_NAME = "yolov8"
CFG.TRAIN_MODEL_SEQUENCE = False
CFG.EPOCHS_OVERRIDE = 100
CFG.RESUME = False
CFG.FORCE_REBUILD_DATASETS = True
CFG.RDD2022_ROOT = "/home/road_damage/datasets/rdd2022/extracted/China_Drone"
CFG.YOLOV8_EVAL_TTA = False


MODEL_SPECS: Dict[str, ModelSpec] = {
    "yolov4": ModelSpec(
        model_name="yolov4",
        family="darknet",
        display_name="YOLOv4-tiny",
        repo_name="darknet",
        repo_url="https://github.com/AlexeyAB/darknet.git",
        repo_ref="master",
        pretrained_weights="https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29",
        batch_size=32,
        epochs=300,
        img_size=640,
        notes="Paper refers to YOLOv4-tiny as the initial basis. Darknet is the most faithful Kaggle-friendly path.",
    ),
    "yolov5": ModelSpec(
        model_name="yolov5",
        family="yolov5",
        display_name="YOLOv5x v7.0",
        repo_name="yolov5",
        repo_url="https://github.com/ultralytics/yolov5.git",
        repo_ref="v7.0",
        pretrained_weights="yolov5x.pt",
        train_script="train.py",
        detect_script="detect.py",
        hyp_file="data/hyps/hyp.scratch-high.yaml",
        batch_size=16,
        epochs=300,
        img_size=640,
        notes="The paper reports YOLOv5 v7.0 and explicitly mentions YOLOv5x.",
    ),
    "yolov5_tph": ModelSpec(
        model_name="yolov5_tph",
        family="tph",
        display_name="TPH-YOLOv5",
        repo_name="tph-yolov5",
        repo_url="https://github.com/cv516Buaa/tph-yolov5.git",
        repo_ref="main",
        pretrained_weights="yolov5l.pt",
        cfg="models/yolov5l-xs-tph.yaml",
        train_script="train.py",
        detect_script="detect.py",
        hyp_file="data/hyps/hyp.VisDrone.yaml",
        batch_size=8,
        epochs=212,
        optimizer="Adam",
        img_size=640,
        notes="The official public TPH implementation exposes yolov5l-xs-tph; the paper does not publish a road-damage-specific TPH config.",
    ),
    "yolov7": ModelSpec(
        model_name="yolov7",
        family="yolov7",
        display_name="YOLOv7-W6",
        repo_name="yolov7",
        repo_url="https://github.com/WongKinYiu/yolov7.git",
        repo_ref="main",
        pretrained_weights="yolov7-w6_training.pt",
        cfg="cfg/training/yolov7-w6-custom.yaml",
        train_script="train_aux.py",
        detect_script="detect.py",
        hyp_file="data/hyp.scratch.custom.yaml",
        batch_size=16,
        epochs=300,
        img_size=640,
        notes="The paper says YOLOv7x-W6, which is not an official upstream identifier. This notebook uses official yolov7-w6 as the closest released counterpart.",
    ),
    "yolov8": ModelSpec(
        model_name="yolov8",
        family="ultralytics",
        display_name="YOLOv8x",
        repo_name=None,
        repo_url=None,
        repo_ref=None,
        pretrained_weights="yolov8x.pt",
        batch_size=8,
        epochs=100,
        img_size=640,
        notes="YOLOv8 is the extension model requested for the notebook. The default Kaggle-friendly setup uses a single GPU, batch size 8, and 100 epochs.",
    ),
}


def ensure_dirs(cfg: Config):
    for path_str in [cfg.ROOT_OUT, cfg.REPOS_DIR, cfg.DATASETS_DIR, cfg.EXPERIMENTS_DIR, cfg.SUMMARIES_DIR, cfg.CONFIGS_DIR, cfg.LOGS_DIR]:
        Path(path_str).mkdir(parents=True, exist_ok=True)


ensure_dirs(CFG)


def apply_runtime_overrides(cfg: Config) -> Config:
    cfg = Config(**asdict(cfg))
    if cfg.DRY_RUN or cfg.SANITY_MODE:
        cfg.WORKERS = 2
    return cfg


CFG = apply_runtime_overrides(CFG)
GPU_COUNT = torch.cuda.device_count()
GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)]
DEVICE_STRING = ",".join(str(i) for i in range(GPU_COUNT)) if GPU_COUNT else "cpu"


def primary_gpu_name() -> str:
    return GPU_NAMES[0] if GPU_NAMES else "cpu"


def primary_gpu_memory_gb() -> float:
    if not GPU_COUNT:
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / float(1024 ** 3)


PRIMARY_GPU_NAME = primary_gpu_name()
PRIMARY_GPU_MEMORY_GB = primary_gpu_memory_gb()


def resolved_worker_count(default_workers: int) -> int:
    if CFG.DRY_RUN or CFG.SANITY_MODE:
        return 2
    cpu_count = os.cpu_count() or default_workers
    if not GPU_COUNT:
        return max(2, min(default_workers, cpu_count // 2))
    if PRIMARY_GPU_MEMORY_GB >= 40:
        return max(default_workers, min(16, max(8, cpu_count // 2)))
    return default_workers


def recommended_yolov8_img_size(default_img_size: int) -> int:
    if not GPU_COUNT or not CFG.YOLOV8_ACCURACY_MODE:
        return default_img_size
    if PRIMARY_GPU_MEMORY_GB >= 44:
        return max(default_img_size, CFG.YOLOV8_ACCURACY_IMGSZ)
    if PRIMARY_GPU_MEMORY_GB >= 30:
        return max(default_img_size, min(CFG.YOLOV8_ACCURACY_IMGSZ, 896))
    return default_img_size


def recommended_yolov8_batch_size(default_batch: int, img_size: int, accuracy_mode: bool = False) -> int:
    if not GPU_COUNT:
        return default_batch
    if accuracy_mode:
        if img_size >= 960:
            if PRIMARY_GPU_MEMORY_GB >= 44:
                return 12
            if PRIMARY_GPU_MEMORY_GB >= 30:
                return 8
        if img_size >= 896:
            if PRIMARY_GPU_MEMORY_GB >= 44:
                return 14
            if PRIMARY_GPU_MEMORY_GB >= 30:
                return 10
        if PRIMARY_GPU_MEMORY_GB >= 44:
            return 16
        if PRIMARY_GPU_MEMORY_GB >= 30:
            return 12
        return default_batch
    if img_size != 640:
        return default_batch
    gpu_name = PRIMARY_GPU_NAME.lower().replace(" ", "")
    if "rtx6000ada" in gpu_name:
        return max(default_batch, 32)
    if "a6000" in gpu_name:
        return max(default_batch, 24)
    if PRIMARY_GPU_MEMORY_GB >= 44:
        return max(default_batch, 24)
    if PRIMARY_GPU_MEMORY_GB >= 30:
        return max(default_batch, 16)
    if PRIMARY_GPU_MEMORY_GB >= 22:
        return max(default_batch, 12)
    return default_batch


def yolov8_batch_candidates(start_batch: int) -> List[int]:
    candidates = []
    current = max(1, int(start_batch))
    while current >= 8:
        if current not in candidates:
            candidates.append(current)
        if current == 8:
            break
        current = max(8, current - 8 if current > 16 else current // 2)
    if 8 not in candidates:
        candidates.append(8)
    return candidates

print("IS_KAGGLE:", IS_KAGGLE)
print("GPU_COUNT:", GPU_COUNT)
print("GPU_NAMES:", GPU_NAMES)
print("DEVICE_STRING:", DEVICE_STRING)
print("PRIMARY_GPU_NAME:", PRIMARY_GPU_NAME)
print("PRIMARY_GPU_MEMORY_GB:", round(PRIMARY_GPU_MEMORY_GB, 2))
print("ROOT_OUT:", CFG.ROOT_OUT)

# %% [markdown]
# ## 2. Assumptions And Research Notes
#
# The paper is not perfectly specified. The notebook keeps an explicit assumptions log and writes it to disk.

# %%
ASSUMPTIONS = [
    {
        "id": "spain_rebuild_from_original_repo",
        "status": "active",
        "assumption": "The unavailable Roboflow Spain export is reconstructed from the original luisaugustos/Pothole-Recognition repository referenced by the paper.",
        "why": "The paper explicitly references that repository when describing the Spain UAV dataset.",
    },
    {
        "id": "spain_label_mapping_if_numeric_only",
        "status": "active",
        "assumption": "If the Spain repository exposes only numeric class IDs without a class file, the majority class is mapped to D40 and the minority class is mapped to D00.",
        "why": "Paper Table 1 reports 3480 Spain annotations for D40 and 327 for D00.",
    },
    {
        "id": "yolov7_variant_selection",
        "status": "active",
        "assumption": "The paper string 'YOLOv7x-W6' is treated as the official upstream YOLOv7-W6 release.",
        "why": "No official upstream model is named 'yolov7x-w6'.",
    },
    {
        "id": "paper_count_inconsistency_handling",
        "status": "active",
        "assumption": "Raw merged image counts and augmented training-view counts are tracked separately and never silently collapsed into one number.",
        "why": "The paper reports 2893 merged images, 4000 training images, and 4873 split images, which are not fully self-consistent without an augmentation stage.",
    },
    {
        "id": "merged_train_offline_augmentation",
        "status": "active",
        "assumption": "Offline train augmentation after the raw train split uses paper-described rotation in the range -15° to +15° to build the augmented train view.",
        "why": "The paper explicitly states rotation in this range for the merged dataset.",
    },
    {
        "id": "online_aug_is_mild",
        "status": "active",
        "assumption": "Online training augmentation is kept mild to avoid the geometric distortions criticized by the paper figures for default YOLO augmentation.",
        "why": "The paper notes that default parameters can distort the road-damage imagery.",
    },
    {
        "id": "tph_model_choice",
        "status": "active",
        "assumption": "The official TPH-YOLOv5 public config models/yolov5l-xs-tph.yaml is used.",
        "why": "It is the official public implementation and the paper does not publish an alternate road-damage-specific TPH config.",
    },
]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


save_json(ASSUMPTIONS, Path(CFG.SUMMARIES_DIR) / "assumptions.json")
pd.DataFrame(ASSUMPTIONS).to_csv(Path(CFG.SUMMARIES_DIR) / "assumptions.csv", index=False)
print(pd.DataFrame(ASSUMPTIONS))

# %% [markdown]
# ## 3. Common Utilities

# %%
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.SEED)


def env_with_utf8(extra: Optional[Dict[str, str]] = None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    log_path: Optional[Path] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
):
    cwd = Path(cwd) if cwd else None
    print("Running:", " ".join(map(str, cmd)))
    if log_path is None:
        completed = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env_with_utf8(env), check=check)
        return completed
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env_with_utf8(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            logf.write(line)
        ret = process.wait()
        if check and ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(map(str, cmd))}")
        return ret


def internet_available(host: str = "github.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def is_git_repository(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            env=env_with_utf8(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def discard_broken_repo_cache(repo_dir: Path):
    repos_root = Path(CFG.REPOS_DIR).resolve()
    resolved_repo_dir = repo_dir.resolve()
    if resolved_repo_dir == repos_root or repos_root not in resolved_repo_dir.parents:
        raise RuntimeError(f"Refusing to delete repo cache outside REPOS_DIR: {repo_dir}")
    shutil.rmtree(repo_dir)


def git_clone_or_checkout(repo_url: str, repo_dir: Path, ref: Optional[str] = None):
    if repo_dir.exists() and not repo_dir.is_dir():
        raise RuntimeError(f"Expected a repo directory, found a file instead: {repo_dir}")

    repo_has_files = repo_dir.exists() and any(repo_dir.iterdir())
    repo_is_git = repo_has_files and is_git_repository(repo_dir)

    if repo_is_git:
        print(f"Using cached git repo: {repo_dir}")
    elif repo_has_files:
        if internet_available():
            print(f"Cached repo is missing git metadata, re-cloning: {repo_dir}")
            discard_broken_repo_cache(repo_dir)
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], log_path=Path(CFG.LOGS_DIR) / f"{repo_dir.name}_clone.log")
            repo_is_git = True
        else:
            print(f"Using cached extracted snapshot (no .git): {repo_dir}")
            if ref:
                print(f"Skipping checkout of {ref!r} because internet is unavailable and {repo_dir} is not a git repository.")
            return repo_dir
    else:
        if not internet_available():
            raise RuntimeError(f"Internet access is required to clone {repo_url}. Enable Kaggle Internet or pre-cache the repo.")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], log_path=Path(CFG.LOGS_DIR) / f"{repo_dir.name}_clone.log")
        repo_is_git = True
    if ref and repo_is_git:
        with contextlib.suppress(Exception):
            run_cmd(["git", "fetch", "--tags", "--all"], cwd=repo_dir, log_path=Path(CFG.LOGS_DIR) / f"{repo_dir.name}_fetch.log", check=False)
        run_cmd(["git", "checkout", ref], cwd=repo_dir, log_path=Path(CFG.LOGS_DIR) / f"{repo_dir.name}_checkout.log")
    return repo_dir


def detect_rdd2022_root() -> Path:
    if CFG.RDD2022_ROOT:
        path = Path(CFG.RDD2022_ROOT)
        if path.exists():
            return path
        raise FileNotFoundError(f"CFG.RDD2022_ROOT does not exist: {path}")
    if not KAGGLE_INPUT.exists():
        raise FileNotFoundError("Kaggle input directory was not found.")
    candidates = []
    for p in KAGGLE_INPUT.rglob("*"):
        if p.is_dir() and p.name.lower() in {"china_drone", "china-uav", "china_uav"}:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            "Could not find RDD2022 China_Drone inside /kaggle/input. Attach the RDD2022 dataset to the notebook first."
        )
    best = sorted(candidates, key=lambda x: len(x.parts))[0]
    return best


RDD2022_ROOT = detect_rdd2022_root()
print("RDD2022_ROOT:", RDD2022_ROOT)


def maybe_override_for_dry_run(samples: List[dict], source_name: str) -> List[dict]:
    if not (CFG.DRY_RUN or CFG.SANITY_MODE):
        return samples
    capped = samples[: CFG.DRY_RUN_MAX_IMAGES_PER_SOURCE]
    print(f"[DRY RUN] {source_name}: keeping {len(capped)} of {len(samples)} samples")
    return capped


def read_image_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return np.array(im)
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"Failed to read image {path}: {e}") from e


def write_image_rgb(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=95)


def xyxy_to_yolo(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0), w - 1)
    x2 = min(max(x2, 0), w - 1)
    y1 = min(max(y1, 0), h - 1)
    y2 = min(max(y2, 0), h - 1)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return [xc / w, yc / h, bw / w, bh / h]


def yolo_to_xyxy(box: Sequence[float], w: int, h: int) -> List[float]:
    xc, yc, bw, bh = box
    bw *= w
    bh *= h
    xc *= w
    yc *= h
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    return [x1, y1, x2, y2]


def clip_box_xyxy(box: Sequence[float], w: int, h: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, w))
    x2 = float(np.clip(x2, 0, w))
    y1 = float(np.clip(y1, 0, h))
    y2 = float(np.clip(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = box_area(box1) + box_area(box2) - inter
    return inter / max(union, 1e-9)


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").replace("-", " ").split())


CLASS_ALIASES = {
    "D00": {"d00", "longitudinal crack", "longitudinal cracks", "wheel mark", "crack"},
    "D10": {"d10", "transverse crack", "transverse cracks", "lateral crack", "equal interval"},
    "D20": {"d20", "alligator crack", "alligator cracks", "crocodile crack", "reticular crack", "block-like crack"},
    "D40": {"d40", "pothole", "potholes", "rutting", "bump", "separation", "other corruption"},
    "Repair": {"repair", "repaired area", "recheck", "patch", "patched area"},
    "Block Crack": {"block crack", "block cracks", "blockcrack"},
}


def canonicalize_label(raw_name: str, detail: str = "", source: str = "") -> Optional[str]:
    name_norm = normalize_text(raw_name)
    detail_norm = normalize_text(detail)
    merged = f"{name_norm} {detail_norm}".strip()

    if name_norm in {"d00", "d01"}:
        return "D00"
    if name_norm in {"d10", "d11"}:
        return "D10"
    if name_norm == "d20":
        return "D20"
    if name_norm == "d40":
        if "repair" in merged:
            return "Repair"
        if "block" in merged:
            return "Block Crack"
        return "D40"
    if name_norm in {"d43", "d44"}:
        if "repair" in merged:
            return "Repair"
        if "block" in merged:
            return "Block Crack"
        return None

    for canonical, aliases in CLASS_ALIASES.items():
        if name_norm in aliases or detail_norm in aliases or any(alias in merged for alias in aliases):
            return canonical
    return None


def save_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def write_yolo_label(path: Path, labels: List[dict], class_to_idx: Dict[str, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in labels:
            cls = class_to_idx[item["class_name"]]
            xc, yc, bw, bh = item["bbox_yolo"]
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def load_yolo_label(path: Path, with_conf: bool = False) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            expected = 6 if with_conf else 5
            if len(parts) != expected:
                raise ValueError(f"Malformed YOLO row in {path}:{line_num}: {line}")
            values = list(map(float, parts))
            row = {
                "class_id": int(values[0]),
                "bbox_yolo": values[1:5],
            }
            if with_conf:
                row["conf"] = values[5]
            rows.append(row)
    return rows


def validate_yolo_rows(rows: List[dict], num_classes: int, path: Path):
    for idx, row in enumerate(rows, start=1):
        class_id = row["class_id"]
        xc, yc, bw, bh = row["bbox_yolo"]
        assert 0 <= class_id < num_classes, f"Out-of-range class id in {path}:{idx}"
        assert 0 <= xc <= 1 and 0 <= yc <= 1, f"Center out of range in {path}:{idx}"
        assert 0 < bw <= 1 and 0 < bh <= 1, f"Box size out of range in {path}:{idx}"


def resize_direct_keep_yolo_labels(image: np.ndarray, labels_xyxy: List[dict], out_size: int) -> Tuple[np.ndarray, List[dict]]:
    h, w = image.shape[:2]
    resized = cv2.resize(image, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    out_labels = []
    for item in labels_xyxy:
        yolo_box = xyxy_to_yolo(item["bbox_xyxy"], w, h)
        out_labels.append({"class_name": item["class_name"], "bbox_yolo": yolo_box})
    return resized, out_labels


def parse_pascal_voc(xml_path: Path) -> List[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for obj in root.findall(".//object"):
        name = obj.findtext("name", default="")
        detail_candidates = []
        for tag in ["detail", "description", "subtype", "remarks", "comment"]:
            val = obj.findtext(tag, default="")
            if val:
                detail_candidates.append(val)
        detail = " ".join(detail_candidates)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        rows.append({"raw_name": name, "detail": detail, "bbox_xyxy": [xmin, ymin, xmax, ymax]})
    return rows


def discover_image_label_pairs(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    images = {}
    labels = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        stem = path.stem
        if suffix in IMAGE_EXTS:
            images[stem] = path
        elif suffix in {".txt", ".xml"}:
            labels[stem] = path
    return images, labels

# %% [markdown]
# ## 4. Repository Setup

# %%
def prepare_repo_dependencies(model_name: str):
    spec = MODEL_SPECS[model_name]
    if spec.repo_name is None:
        return None
    repo_dir = Path(CFG.REPOS_DIR) / spec.repo_name
    git_clone_or_checkout(spec.repo_url, repo_dir, spec.repo_ref)

    if spec.family == "darknet":
        makefile = repo_dir / "Makefile"
        text = makefile.read_text(encoding="utf-8")
        for old, new in [
            ("GPU=0", "GPU=1"),
            ("CUDNN=0", "CUDNN=1"),
            ("CUDNN_HALF=0", "CUDNN_HALF=1"),
            ("OPENCV=0", "OPENCV=1"),
            ("LIBSO=0", "LIBSO=1"),
        ]:
            text = text.replace(old, new)
        makefile.write_text(text, encoding="utf-8")
        run_cmd(["make", "-j4"], cwd=repo_dir, log_path=Path(CFG.LOGS_DIR) / "darknet_build.log")
    else:
        if EARLY_IS_KAGGLE:
            repo_extras = {
                "yolov5": [("thop", "thop>=0.1.1"), ("tensorboard", "tensorboard>=2.15")],
                "tph-yolov5": [("thop", "thop>=0.1.1"), ("tensorboard", "tensorboard>=2.15"), ("einops", "einops>=0.7")],
                "yolov7": [("thop", "thop>=0.1.1"), ("tensorboard", "tensorboard>=2.15")],
            }
            pip_install_if_needed(repo_extras.get(spec.repo_name, []))
        else:
            req = repo_dir / "requirements.txt"
            if req.exists():
                run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], cwd=repo_dir, log_path=Path(CFG.LOGS_DIR) / f"{spec.repo_name}_pip.log")
    return repo_dir


MODELS_TO_PREPARE = CFG.MODEL_SEQUENCE if CFG.TRAIN_MODEL_SEQUENCE else [CFG.MODEL_NAME]
PREPARED_REPOS = {}
for model_name in MODELS_TO_PREPARE:
    PREPARED_REPOS[model_name] = prepare_repo_dependencies(model_name)
print(PREPARED_REPOS)

# %% [markdown]
# ## 5. Spain UAV Dataset Reconstruction

# %%
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CFG.TARGET_CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def infer_spain_txt_mapping(label_paths: List[Path]) -> Dict[int, str]:
    counts = Counter()
    for path in label_paths:
        for row in load_yolo_label(path, with_conf=False):
            counts[row["class_id"]] += 1
    if not counts:
        raise RuntimeError("Could not infer Spain class mapping because no labels were found.")
    unique_ids = sorted(counts)
    if len(unique_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 Spain classes, found {len(unique_ids)}: {counts}")
    ranked = [cid for cid, _ in counts.most_common()]
    mapping = {ranked[0]: "D40", ranked[1]: "D00"}
    print("Inferred Spain numeric mapping from annotation frequency:", mapping, counts)
    return mapping


def collect_spain_original_samples() -> List[dict]:
    repo_dir = Path(CFG.REPOS_DIR) / "Pothole-Recognition"
    git_clone_or_checkout(CFG.SPAIN_REPO_URL, repo_dir, CFG.SPAIN_REPO_REF)
    dataset_dir = repo_dir / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Spain dataset directory not found: {dataset_dir}")

    image_paths, label_paths = discover_image_label_pairs(dataset_dir)
    paired_stems = sorted(set(image_paths).intersection(label_paths))
    if not paired_stems:
        raise RuntimeError("No paired Spain images and labels were found in the Pothole-Recognition repository.")

    numeric_mapping = infer_spain_txt_mapping([label_paths[s] for s in paired_stems if label_paths[s].suffix.lower() == ".txt"])
    samples = []

    for stem in tqdm(paired_stems, desc="Collecting Spain originals"):
        img_path = image_paths[stem]
        lab_path = label_paths[stem]
        img = read_image_rgb(img_path)
        h, w = img.shape[:2]
        labels_xyxy = []

        if lab_path.suffix.lower() == ".txt":
            for row in load_yolo_label(lab_path, with_conf=False):
                cls_name = numeric_mapping[row["class_id"]]
                box = yolo_to_xyxy(row["bbox_yolo"], w, h)
                clipped = clip_box_xyxy(box, w, h)
                if clipped is None:
                    continue
                labels_xyxy.append({"class_name": cls_name, "bbox_xyxy": clipped})
        else:
            for row in parse_pascal_voc(lab_path):
                cls_name = canonicalize_label(row["raw_name"], row["detail"], source="Spain")
                if cls_name is None:
                    continue
                clipped = clip_box_xyxy(row["bbox_xyxy"], w, h)
                if clipped is None:
                    continue
                labels_xyxy.append({"class_name": cls_name, "bbox_xyxy": clipped})

        labels_xyxy = [x for x in labels_xyxy if x["class_name"] in {"D00", "D40"}]
        if not labels_xyxy:
            continue

        samples.append(
            {
                "sample_id": f"spain_raw_{stem}",
                "origin_id": f"spain_raw_{stem}",
                "source": "Spain",
                "image_path": img_path,
                "label_path": lab_path,
                "labels_xyxy": labels_xyxy,
                "width": w,
                "height": h,
                "augmented": False,
            }
        )
    samples = maybe_override_for_dry_run(samples, "Spain raw originals")
    return samples


def center_zoom_transform(image: np.ndarray, labels_xyxy: List[dict], zoom_fraction: float) -> Tuple[np.ndarray, List[dict]]:
    if zoom_fraction <= 0:
        return image.copy(), [dict(x) for x in labels_xyxy]
    h, w = image.shape[:2]
    crop_w = max(16, int(round(w / (1.0 + zoom_fraction))))
    crop_h = max(16, int(round(h / (1.0 + zoom_fraction))))
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    crop = image[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    out_labels = []
    sx = w / crop_w
    sy = h / crop_h
    crop_box = [x1, y1, x2, y2]
    for item in labels_xyxy:
        box = item["bbox_xyxy"]
        ix1 = max(box[0], crop_box[0])
        iy1 = max(box[1], crop_box[1])
        ix2 = min(box[2], crop_box[2])
        iy2 = min(box[3], crop_box[3])
        clipped = clip_box_xyxy([ix1, iy1, ix2, iy2], w, h)
        if clipped is None:
            continue
        clipped_area = box_area(clipped)
        if clipped_area / max(box_area(box), 1e-9) < 0.35:
            continue
        nx1 = (clipped[0] - x1) * sx
        ny1 = (clipped[1] - y1) * sy
        nx2 = (clipped[2] - x1) * sx
        ny2 = (clipped[3] - y1) * sy
        nclipped = clip_box_xyxy([nx1, ny1, nx2, ny2], w, h)
        if nclipped is None:
            continue
        out_labels.append({"class_name": item["class_name"], "bbox_xyxy": nclipped})
    return crop_resized, out_labels


def build_clean_yolo_subset(samples: List[dict], subset_name: str, out_root: Path) -> List[dict]:
    records = []
    image_out_dir = out_root / "images" / subset_name
    label_out_dir = out_root / "labels" / subset_name
    for sample in tqdm(samples, desc=f"Writing {subset_name}"):
        img = read_image_rgb(sample["image_path"])
        labels_xyxy = sample["labels_xyxy"]
        resized_img, labels_yolo = resize_direct_keep_yolo_labels(img, labels_xyxy, CFG.IMG_SIZE)
        image_name = f"{sample['sample_id']}.jpg"
        label_name = f"{sample['sample_id']}.txt"
        img_path = image_out_dir / image_name
        lab_path = label_out_dir / label_name
        write_image_rgb(img_path, resized_img)
        write_yolo_label(lab_path, labels_yolo, CLASS_TO_IDX)
        validate_yolo_rows(load_yolo_label(lab_path, with_conf=False), len(CLASS_TO_IDX), lab_path)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "origin_id": sample["origin_id"],
                "source": sample["source"],
                "split": subset_name,
                "image_path": str(img_path),
                "label_path": str(lab_path),
                "augmented": sample["augmented"],
                "classes_present": sorted({x["class_name"] for x in labels_yolo}),
                "num_boxes": len(labels_yolo),
            }
        )
    return records


def summarize_records(records: List[dict], stage_name: str, out_prefix: Path) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "stage": stage_name,
                "source": r["source"],
                "split": r["split"],
                "sample_id": r["sample_id"],
                "origin_id": r["origin_id"],
                "augmented": r["augmented"],
                "num_boxes": r["num_boxes"],
                "classes_present": "|".join(r["classes_present"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_prefix.with_suffix(".csv"), index=False)
    save_json(rows, out_prefix.with_suffix(".json"))
    return df


def count_labels_from_dataset_dir(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for split in ["raw", "augmented", "train", "val", "test"]:
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        for txt in sorted(label_dir.glob("*.txt")):
            labels = load_yolo_label(txt)
            for row in labels:
                rows.append(
                    {
                        "split": split,
                        "sample_id": txt.stem,
                        "class_id": row["class_id"],
                        "class_name": IDX_TO_CLASS[row["class_id"]],
                    }
                )
    return pd.DataFrame(rows)


def plot_class_distribution(label_df: pd.DataFrame, title: str, out_path: Path):
    if label_df.empty:
        return
    counts = label_df.groupby(["split", "class_name"]).size().reset_index(name="count")
    plt.figure(figsize=(12, 5))
    sns.barplot(data=counts, x="class_name", y="count", hue="split")
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def rebuild_spain_dataset() -> Tuple[Path, pd.DataFrame]:
    out_root = Path(CFG.DATASETS_DIR) / "spain_rebuilt"
    manifest_path = out_root / "manifest.csv"
    if manifest_path.exists() and not CFG.FORCE_REBUILD_DATASETS:
        print("Using cached Spain rebuilt dataset:", out_root)
        return out_root, pd.read_csv(manifest_path)

    if out_root.exists():
        shutil.rmtree(out_root)

    original_samples = collect_spain_original_samples()
    if len(original_samples) == 0:
        raise RuntimeError("No usable Spain samples found.")

    raw_records = build_clean_yolo_subset(original_samples, "raw", out_root)
    target_total = min(CFG.SPAIN_TARGET_TOTAL_IMAGES, len(original_samples) if (CFG.DRY_RUN or CFG.SANITY_MODE) else CFG.SPAIN_TARGET_TOTAL_IMAGES)
    extra_needed = max(0, target_total - len(original_samples))
    rng = np.random.default_rng(CFG.SEED)
    chosen_indices = list(rng.permutation(len(original_samples)))
    aug_samples = []
    for aug_idx in range(extra_needed):
        src = original_samples[chosen_indices[aug_idx % len(chosen_indices)]]
        img = read_image_rgb(src["image_path"])
        zoom = float(rng.uniform(CFG.SPAIN_ZOOM_RANGE[0], CFG.SPAIN_ZOOM_RANGE[1]))
        zoomed_img, zoomed_labels = center_zoom_transform(img, src["labels_xyxy"], zoom)
        if not zoomed_labels:
            continue
        temp_dir = out_root / "_tmp_spain_aug"
        temp_img_path = temp_dir / f"{src['sample_id']}_zoom_{aug_idx:04d}.jpg"
        write_image_rgb(temp_img_path, zoomed_img)
        aug_samples.append(
            {
                "sample_id": f"{src['sample_id']}_zoom_{aug_idx:04d}",
                "origin_id": src["origin_id"],
                "source": "Spain",
                "image_path": temp_img_path,
                "label_path": None,
                "labels_xyxy": zoomed_labels,
                "width": zoomed_img.shape[1],
                "height": zoomed_img.shape[0],
                "augmented": True,
            }
        )

    aug_records = build_clean_yolo_subset(aug_samples, "augmented", out_root) if aug_samples else []
    manifest = pd.DataFrame(raw_records + aug_records)
    manifest.to_csv(manifest_path, index=False)
    summarize_records(raw_records + aug_records, "spain_rebuilt", Path(CFG.SUMMARIES_DIR) / "spain_rebuilt_manifest")

    label_df = count_labels_from_dataset_dir(out_root)
    label_df.to_csv(out_root / "label_summary.csv", index=False)
    plot_class_distribution(label_df, "Spain rebuilt class distribution", out_root / "plots" / "spain_rebuilt_distribution.png")
    return out_root, manifest


SPAIN_DATASET_ROOT, SPAIN_MANIFEST = rebuild_spain_dataset()
print(SPAIN_MANIFEST.head())
print("Spain rebuilt counts:", SPAIN_MANIFEST["split"].value_counts().to_dict())

# %% [markdown]
# ## 6. RDD2022 China UAV Conversion

# %%
def collect_rdd2022_china_samples() -> List[dict]:
    images, labels = discover_image_label_pairs(RDD2022_ROOT)
    china_image_paths = {k: v for k, v in images.items() if "china" in str(v).lower() and "drone" in str(v).lower()}
    china_label_paths = {k: v for k, v in labels.items() if "china" in str(v).lower() and "drone" in str(v).lower()}
    stems = sorted(set(china_image_paths).intersection(china_label_paths))
    if not stems:
        raise RuntimeError("Could not find paired China_Drone image/xml files under the attached RDD2022 dataset.")
    samples = []
    for stem in tqdm(stems, desc="Collecting RDD2022 China_Drone"):
        img_path = china_image_paths[stem]
        lab_path = china_label_paths[stem]
        img = read_image_rgb(img_path)
        h, w = img.shape[:2]
        rows = parse_pascal_voc(lab_path) if lab_path.suffix.lower() == ".xml" else []
        labels_xyxy = []
        for row in rows:
            cls_name = canonicalize_label(row["raw_name"], row["detail"], source="China_Drone")
            if cls_name is None or cls_name not in CFG.TARGET_CLASSES:
                continue
            clipped = clip_box_xyxy(row["bbox_xyxy"], w, h)
            if clipped is None:
                continue
            labels_xyxy.append({"class_name": cls_name, "bbox_xyxy": clipped})
        if not labels_xyxy:
            continue
        samples.append(
            {
                "sample_id": f"china_uav_raw_{stem}",
                "origin_id": f"china_uav_raw_{stem}",
                "source": "China_Drone",
                "image_path": img_path,
                "label_path": lab_path,
                "labels_xyxy": labels_xyxy,
                "width": w,
                "height": h,
                "augmented": False,
            }
        )
    samples = maybe_override_for_dry_run(samples, "RDD2022 China_Drone")
    return samples


def rebuild_china_dataset() -> Tuple[Path, pd.DataFrame]:
    out_root = Path(CFG.DATASETS_DIR) / "china_uav_rebuilt"
    manifest_path = out_root / "manifest.csv"
    if manifest_path.exists() and not CFG.FORCE_REBUILD_DATASETS:
        print("Using cached China rebuilt dataset:", out_root)
        return out_root, pd.read_csv(manifest_path)

    if out_root.exists():
        shutil.rmtree(out_root)

    china_samples = collect_rdd2022_china_samples()
    if not china_samples:
        raise RuntimeError("No usable China_Drone samples were found after canonical class filtering.")
    raw_records = build_clean_yolo_subset(china_samples, "raw", out_root)
    manifest = pd.DataFrame(raw_records)
    manifest.to_csv(manifest_path, index=False)
    summarize_records(raw_records, "china_uav_rebuilt", Path(CFG.SUMMARIES_DIR) / "china_uav_rebuilt_manifest")

    label_df = count_labels_from_dataset_dir(out_root)
    label_df.to_csv(out_root / "label_summary.csv", index=False)
    plot_class_distribution(label_df, "China UAV rebuilt class distribution", out_root / "plots" / "china_uav_distribution.png")
    return out_root, manifest


CHINA_DATASET_ROOT, CHINA_MANIFEST = rebuild_china_dataset()
print(CHINA_MANIFEST.head())
print("China rebuilt counts:", CHINA_MANIFEST["split"].value_counts().to_dict())

# %% [markdown]
# ## 7. Merge, Validation Checks, And Split Build

# %%
def manifest_to_samples_for_merge(dataset_root: Path, manifest_df: pd.DataFrame, include_splits: Sequence[str]) -> List[dict]:
    samples = []
    for _, row in manifest_df.iterrows():
        if row["split"] not in include_splits:
            continue
        lab_path = Path(row["label_path"])
        img_path = Path(row["image_path"])
        labels = load_yolo_label(lab_path)
        validate_yolo_rows(labels, len(CLASS_TO_IDX), lab_path)
        samples.append(
            {
                "sample_id": row["sample_id"],
                "origin_id": row["origin_id"],
                "source": row["source"],
                "image_path": img_path,
                "label_path": lab_path,
                "labels": labels,
                "classes_present": sorted({IDX_TO_CLASS[x["class_id"]] for x in labels}),
                "augmented": bool(row["augmented"]),
            }
        )
    return samples


def greedy_multilabel_split(samples: List[dict], ratios=(0.7, 0.2, 0.1), seed: int = 42) -> Dict[str, List[dict]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    samples = samples.copy()
    rng.shuffle(samples)
    n = len(samples)
    targets = {
        "train": int(round(ratios[0] * n)),
        "val": int(round(ratios[1] * n)),
        "test": n,
    }
    targets["test"] = n - targets["train"] - targets["val"]

    global_cls_counts = Counter()
    for s in samples:
        for cls in set(s["classes_present"]):
            global_cls_counts[cls] += 1
    source_counts = Counter([s["source"] for s in samples])

    def rarity_score(sample):
        cls_score = sum(1.0 / max(global_cls_counts[c], 1) for c in set(sample["classes_present"]))
        src_score = 1.0 / max(source_counts[sample["source"]], 1)
        return cls_score + 0.25 * src_score

    samples_sorted = sorted(samples, key=rarity_score, reverse=True)
    splits = {"train": [], "val": [], "test": []}
    split_cls = {k: Counter() for k in splits}
    split_src = {k: Counter() for k in splits}

    for sample in samples_sorted:
        best_split = None
        best_score = None
        for split_name in ["train", "val", "test"]:
            if len(splits[split_name]) >= targets[split_name]:
                continue
            deficit = targets[split_name] - len(splits[split_name])
            cls_penalty = sum(split_cls[split_name][c] / max(global_cls_counts[c], 1) for c in set(sample["classes_present"]))
            src_penalty = split_src[split_name][sample["source"]] / max(source_counts[sample["source"]], 1)
            score = (-deficit, cls_penalty + 0.2 * src_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name
        if best_split is None:
            best_split = min(splits, key=lambda k: len(splits[k]))
        splits[best_split].append(sample)
        for c in set(sample["classes_present"]):
            split_cls[best_split][c] += 1
        split_src[best_split][sample["source"]] += 1
    return splits


def split_class_presence(splits: Dict[str, List[dict]]) -> Dict[str, Counter]:
    split_cls = {split_name: Counter() for split_name in splits}
    for split_name, split_samples in splits.items():
        for sample in split_samples:
            for cls in set(sample["classes_present"]):
                split_cls[split_name][cls] += 1
    return split_cls


def split_min_image_requirement(split_name: str, class_name: str) -> int:
    return int(CFG.MIN_SPLIT_IMAGES_PER_CLASS.get(split_name, {}).get(class_name, 1))


def ensure_split_class_coverage(
    splits: Dict[str, List[dict]], required_splits: Sequence[str] = ("train", "val", "test")
) -> Tuple[Dict[str, List[dict]], List[dict], List[dict]]:
    splits = {split_name: list(split_samples) for split_name, split_samples in splits.items()}
    split_cls = split_class_presence(splits)
    global_cls = Counter()
    for split_samples in splits.values():
        for sample in split_samples:
            for cls in set(sample["classes_present"]):
                global_cls[cls] += 1

    def donor_cost(sample: dict, donor_name: str) -> Tuple[int, float, int, str]:
        donor_counts = split_cls[donor_name]
        classes = set(sample["classes_present"])
        hard_breaks = sum(1 for cls in classes if donor_counts[cls] <= split_min_image_requirement(donor_name, cls))
        scarcity_cost = sum(
            1.0 / max(donor_counts[cls] - split_min_image_requirement(donor_name, cls), 1) for cls in classes
        )
        return hard_breaks, scarcity_cost, len(classes), sample["sample_id"]

    moves = []
    warnings = []
    warning_keys = set()
    max_iterations = max(
        1,
        sum(max(split_min_image_requirement(split_name, cls), 1) for split_name in required_splits for cls in CFG.TARGET_CLASSES) * 4,
    )
    for _ in range(max_iterations):
        changed = False
        for cls in CFG.TARGET_CLASSES:
            required_total = sum(split_min_image_requirement(split_name, cls) for split_name in required_splits)
            if global_cls[cls] < required_total:
                key = ("insufficient_global_support", cls)
                if key not in warning_keys:
                    warnings.append(
                        {
                            "class_name": cls,
                            "status": "insufficient_global_support",
                            "global_image_count": int(global_cls[cls]),
                            "required_total_images": int(required_total),
                            "required_splits": list(required_splits),
                        }
                    )
                    warning_keys.add(key)
            for recipient_name in required_splits:
                target_count = split_min_image_requirement(recipient_name, cls)
                if split_cls[recipient_name][cls] >= target_count:
                    continue
                donor_names = sorted(
                    [name for name in required_splits if name != recipient_name],
                    key=lambda name: (
                        0 if name == "train" else 1,
                        -(split_cls[name][cls] - split_min_image_requirement(name, cls)),
                        -split_cls[name][cls],
                        -len(splits[name]),
                    ),
                )
                donor_name = None
                donor_sample = None
                for candidate_name in donor_names:
                    candidates = [s for s in splits[candidate_name] if cls in set(s["classes_present"])]
                    if not candidates:
                        continue
                    safe_candidates = [s for s in candidates if donor_cost(s, candidate_name)[0] == 0]
                    donor_pool = safe_candidates or candidates
                    donor_sample = min(donor_pool, key=lambda sample: donor_cost(sample, candidate_name))
                    donor_name = candidate_name
                    break
                if donor_sample is None or donor_name is None:
                    key = ("no_movable_sample_found", cls, recipient_name)
                    if key not in warning_keys:
                        warnings.append(
                            {
                                "class_name": cls,
                                "status": "no_movable_sample_found",
                                "missing_split": recipient_name,
                                "target_count": int(target_count),
                                "current_count": int(split_cls[recipient_name][cls]),
                            }
                        )
                        warning_keys.add(key)
                    continue
                splits[donor_name].remove(donor_sample)
                splits[recipient_name].append(donor_sample)
                for moved_cls in set(donor_sample["classes_present"]):
                    split_cls[donor_name][moved_cls] -= 1
                    split_cls[recipient_name][moved_cls] += 1
                moves.append(
                    {
                        "class_name": cls,
                        "sample_id": donor_sample["sample_id"],
                        "from_split": donor_name,
                        "to_split": recipient_name,
                        "sample_classes": sorted(set(donor_sample["classes_present"])),
                        "target_count": int(target_count),
                        "new_recipient_count": int(split_cls[recipient_name][cls]),
                    }
                )
                changed = True
        if not changed:
            break
    return splits, moves, warnings


def split_class_presence_dataframe(splits: Dict[str, List[dict]]) -> pd.DataFrame:
    split_cls = split_class_presence(splits)
    rows = []
    for split_name in ["train", "val", "test"]:
        for cls in CFG.TARGET_CLASSES:
            rows.append(
                {
                    "split": split_name,
                    "class_name": cls,
                    "num_images_with_class": int(split_cls.get(split_name, Counter()).get(cls, 0)),
                }
            )
    return pd.DataFrame(rows)

def _sanitize_yolo_box(box: Sequence[float], eps: float = 1e-6) -> Optional[List[float]]:
    vals = np.array(box, dtype=float)
    if vals.shape[0] != 4 or not np.isfinite(vals).all():
        return None

    xc, yc, bw, bh = vals.tolist()
    if bw <= eps or bh <= eps:
        return None

    x1 = float(np.clip(xc - bw / 2.0, 0.0, 1.0))
    y1 = float(np.clip(yc - bh / 2.0, 0.0, 1.0))
    x2 = float(np.clip(xc + bw / 2.0, 0.0, 1.0))
    y2 = float(np.clip(yc + bh / 2.0, 0.0, 1.0))

    bw2 = x2 - x1
    bh2 = y2 - y1
    if bw2 <= eps or bh2 <= eps:
        return None

    xc2 = (x1 + x2) / 2.0
    yc2 = (y1 + y2) / 2.0
    return [float(xc2), float(yc2), float(bw2), float(bh2)]


SYNTH_TRAIN_AUG = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ],
            p=0.10,
        ),
        A.Rotate(
            limit=CFG.MERGED_ROTATION_RANGE,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            crop_border=False,
            p=0.85,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25,
        clip=True,
        filter_invalid_bboxes=True,
        check_each_transform=False,
    ),
)


def create_rotated_augmented_train_samples(train_samples: List[dict], extra_needed: int, out_root: Path) -> List[dict]:
    if extra_needed <= 0:
        return []
    rng = np.random.default_rng(CFG.SEED)
    label_counts = Counter()
    class_image_counts = Counter()
    for s in train_samples:
        sample_classes = set(s["classes_present"])
        for c in sample_classes:
            label_counts[c] += 1
            class_image_counts[c] += 1
    dominant_class_image_count = max(class_image_counts.values(), default=0)
    weights = np.array([sum(1.0 / max(label_counts[c], 1) for c in set(s["classes_present"])) for s in train_samples], dtype=float)
    weights = weights / weights.sum()
    aug_samples = []
    temp_aug_dir = out_root / "_tmp_train_aug"
    temp_aug_dir.mkdir(parents=True, exist_ok=True)

    def build_augmented_sample(src: dict, out_index: int, required_class: Optional[str] = None) -> Optional[dict]:
        img = read_image_rgb(Path(src["image_path"]))
        rows = load_yolo_label(Path(src["label_path"]))
        bboxes = []
        labels = []
        for row in rows:
            clean_box = _sanitize_yolo_box(row["bbox_yolo"])
            if clean_box is None:
                continue
            bboxes.append(clean_box)
            labels.append(IDX_TO_CLASS[row["class_id"]])

        if not bboxes:
            return None

        transformed = SYNTH_TRAIN_AUG(image=img, bboxes=bboxes, class_labels=labels)
        if not transformed["bboxes"]:
            return None

        aug_rows = []
        for box, cls in zip(transformed["bboxes"], transformed["class_labels"]):
            clean_box = _sanitize_yolo_box(box)
            if clean_box is None:
                continue
            aug_rows.append({"class_name": cls, "bbox_yolo": clean_box})
        if not aug_rows:
            return None
        if required_class is not None and required_class not in {row["class_name"] for row in aug_rows}:
            return None

        image_path = temp_aug_dir / f"{src['sample_id']}_rot_{out_index:05d}.jpg"
        write_image_rgb(image_path, transformed["image"])
        return {
            "sample_id": f"{src['sample_id']}_rot_{out_index:05d}",
            "origin_id": src["origin_id"],
            "source": src["source"],
            "split": "train",
            "image_path": image_path,
            "label_path": None,
            "labels_aug": aug_rows,
            "classes_present": sorted({row["class_name"] for row in aug_rows}),
            "augmented": True,
        }

    rare_class_schedule = []
    for cls_name, target_count in CFG.MIN_TRAIN_IMAGE_PRESENCE_AFTER_AUG.items():
        ratio_target = CFG.MIN_TRAIN_CLASS_RATIO_AFTER_AUG.get(cls_name)
        if ratio_target is not None and dominant_class_image_count > 0:
            target_count = max(int(target_count), int(math.ceil(dominant_class_image_count * float(ratio_target))))
        current_count = class_image_counts.get(cls_name, 0)
        needed = max(0, int(target_count) - int(current_count))
        rare_class_schedule.extend([cls_name] * needed)
    if rare_class_schedule:
        print("Targeted train augmentation schedule:", dict(Counter(rare_class_schedule)))

    attempts = 0
    max_attempts = max(extra_needed * 15, 5000)
    pbar = tqdm(total=extra_needed, desc="Creating offline train augmentation")
    targeted_index = 0
    while len(aug_samples) < extra_needed and targeted_index < len(rare_class_schedule) and attempts < max_attempts:
        target_class = rare_class_schedule[targeted_index]
        candidate_indices = [i for i, sample in enumerate(train_samples) if target_class in set(sample["classes_present"])]
        if not candidate_indices:
            targeted_index += 1
            continue
        candidate_weights = np.array([weights[i] for i in candidate_indices], dtype=float)
        candidate_weights = candidate_weights / candidate_weights.sum()
        attempts += 1
        src = train_samples[int(rng.choice(candidate_indices, p=candidate_weights))]
        try:
            aug_sample = build_augmented_sample(src, len(aug_samples), required_class=target_class)
        except Exception as e:
            if attempts <= 10:
                print(f"Skipping targeted augmentation for {target_class} from {src['sample_id']}: {repr(e)}")
            continue
        if aug_sample is None:
            continue
        aug_samples.append(aug_sample)
        for cls in set(aug_sample["classes_present"]):
            class_image_counts[cls] += 1
        targeted_index += 1
        pbar.update(1)

    while len(aug_samples) < extra_needed and attempts < max_attempts:
        attempts += 1
        src = train_samples[int(rng.choice(len(train_samples), p=weights))]
        try:
            aug_sample = build_augmented_sample(src, len(aug_samples))
        except Exception as e:
            if attempts <= 10:
                print(f"Skipping bad augmentation sample {src['sample_id']}: {repr(e)}")
            continue
        if aug_sample is None:
            continue
        aug_samples.append(aug_sample)
        for cls in set(aug_sample["classes_present"]):
            class_image_counts[cls] += 1
        pbar.update(1)
    pbar.close()
    print(f"Offline augmentation created {len(aug_samples)} samples after {attempts} attempts.")
    return aug_samples


def write_final_merged_dataset() -> Tuple[Path, pd.DataFrame]:
    out_root = Path(CFG.DATASETS_DIR) / "merged_uav_paper_style"
    manifest_path = out_root / "manifest.csv"
    if manifest_path.exists() and not CFG.FORCE_REBUILD_DATASETS:
        print("Using cached merged dataset:", out_root)
        return out_root, pd.read_csv(manifest_path)

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    spain_samples = manifest_to_samples_for_merge(SPAIN_DATASET_ROOT, SPAIN_MANIFEST, include_splits=["raw", "augmented"])
    china_samples = manifest_to_samples_for_merge(CHINA_DATASET_ROOT, CHINA_MANIFEST, include_splits=["raw"])
    merged_raw_samples = spain_samples + china_samples
    print("Merged raw sample count:", len(merged_raw_samples))

    raw_splits = greedy_multilabel_split(merged_raw_samples, CFG.SPLIT_RATIOS, CFG.SEED)
    raw_splits, split_repair_moves, split_repair_warnings = ensure_split_class_coverage(raw_splits)
    coverage_df = split_class_presence_dataframe(raw_splits)
    coverage_df.to_csv(out_root / "raw_split_class_presence.csv", index=False)
    save_json(split_repair_moves, out_root / "raw_split_repair_moves.json")
    save_json(split_repair_warnings, out_root / "raw_split_repair_warnings.json")
    print("Raw split class coverage after repair:")
    print(coverage_df.pivot(index="split", columns="class_name", values="num_images_with_class").fillna(0).astype(int))
    raw_train = raw_splits["train"]
    raw_val = raw_splits["val"]
    raw_test = raw_splits["test"]

    target_augmented_train = min(CFG.TARGET_AUGMENTED_TRAIN_IMAGES, len(raw_train) if (CFG.DRY_RUN or CFG.SANITY_MODE) else CFG.TARGET_AUGMENTED_TRAIN_IMAGES)
    extra_needed = max(0, target_augmented_train - len(raw_train))
    train_aug_samples = create_rotated_augmented_train_samples(raw_train, extra_needed, out_root)

    def write_split_samples(split_name: str, samples: List[dict]):
        records = []
        img_dir = out_root / "images" / split_name
        lab_dir = out_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(samples, desc=f"Final write {split_name}"):
            src_img_path = Path(sample["image_path"])
            dst_img_path = img_dir / f"{sample['sample_id']}.jpg"
            if sample.get("labels_aug") is not None:
                shutil.copy2(src_img_path, dst_img_path)
                labels_to_write = sample["labels_aug"]
            else:
                shutil.copy2(src_img_path, dst_img_path)
                raw_rows = load_yolo_label(Path(sample["label_path"]))
                labels_to_write = [{"class_name": IDX_TO_CLASS[row["class_id"]], "bbox_yolo": row["bbox_yolo"]} for row in raw_rows]
            dst_lab_path = lab_dir / f"{sample['sample_id']}.txt"
            write_yolo_label(dst_lab_path, labels_to_write, CLASS_TO_IDX)
            validate_yolo_rows(load_yolo_label(dst_lab_path), len(CLASS_TO_IDX), dst_lab_path)
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "origin_id": sample["origin_id"],
                    "source": sample["source"],
                    "split": split_name,
                    "image_path": str(dst_img_path),
                    "label_path": str(dst_lab_path),
                    "augmented": bool(sample["augmented"]),
                    "num_boxes": len(labels_to_write),
                    "classes_present": sorted({row["class_name"] for row in labels_to_write}),
                }
            )
        return records

    final_records = []
    final_records += write_split_samples("train", raw_train)
    final_records += write_split_samples("train", train_aug_samples)
    final_records += write_split_samples("val", raw_val)
    final_records += write_split_samples("test", raw_test)

    manifest = pd.DataFrame(final_records)
    manifest.to_csv(manifest_path, index=False)

    dataset_yaml = {
        "path": str(out_root),
        "train": str(out_root / "images" / "train"),
        "val": str(out_root / "images" / "val"),
        "test": str(out_root / "images" / "test"),
        "nc": len(CFG.TARGET_CLASSES),
        "names": CFG.TARGET_CLASSES,
    }
    save_yaml(dataset_yaml, out_root / "data.yaml")

    summary_rows = []
    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name]
        summary_rows.append(
            {
                "split": split_name,
                "num_images": int(len(split_df)),
                "num_augmented_images": int(split_df["augmented"].sum()),
                "num_raw_images": int((~split_df["augmented"]).sum()),
                "num_labels": int(split_df["num_boxes"].sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_root / "split_summary.csv", index=False)
    summarize_records(final_records, "merged_uav_final", Path(CFG.SUMMARIES_DIR) / "merged_uav_final_manifest")

    label_df = count_labels_from_dataset_dir(out_root)
    label_df.to_csv(out_root / "label_summary.csv", index=False)
    plot_class_distribution(label_df, "Merged UAV class distribution", out_root / "plots" / "merged_distribution.png")

    paper_vs_actual = {
        "paper_spain_raw_images": CFG.SPAIN_TARGET_RAW_IMAGES,
        "paper_spain_total_images": CFG.SPAIN_TARGET_TOTAL_IMAGES,
        "paper_target_augmented_train_images": CFG.TARGET_AUGMENTED_TRAIN_IMAGES,
        "target_classes": CFG.TARGET_CLASSES,
        "actual_spain_raw_images": int(len(SPAIN_MANIFEST[SPAIN_MANIFEST["split"] == "raw"])),
        "actual_spain_total_images": int(len(SPAIN_MANIFEST)),
        "actual_merged_train_images": int((manifest["split"] == "train").sum()),
        "actual_merged_val_images": int((manifest["split"] == "val").sum()),
        "actual_merged_test_images": int((manifest["split"] == "test").sum()),
    }
    save_json(paper_vs_actual, out_root / "paper_vs_actual_counts.json")
    return out_root, manifest


MERGED_DATASET_ROOT, MERGED_MANIFEST = write_final_merged_dataset()
print(MERGED_MANIFEST.groupby(["split", "source", "augmented"]).size())

# %% [markdown]
# ## 8. Dataset Validation And Visualization

# %%
def run_dataset_checks(dataset_root: Path) -> pd.DataFrame:
    issues = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_root / "images" / split
        lab_dir = dataset_root / "labels" / split
        label_files = sorted(lab_dir.glob("*.txt"))
        if not img_dir.exists() or not lab_dir.exists():
            issues.append({"severity": "error", "split": split, "message": "Missing split directory"})
            continue
        for lab_path in tqdm(label_files, desc=f"Validating {split}"):
            img_candidates = [img_dir / f"{lab_path.stem}{ext}" for ext in [".jpg", ".jpeg", ".png"]]
            img_path = next((p for p in img_candidates if p.exists()), None)
            if img_path is None:
                issues.append({"severity": "error", "split": split, "message": f"Missing image for {lab_path.name}"})
                continue
            try:
                img = read_image_rgb(img_path)
            except Exception as e:
                issues.append({"severity": "error", "split": split, "message": str(e)})
                continue
            h, w = img.shape[:2]
            if (h, w) != (CFG.IMG_SIZE, CFG.IMG_SIZE):
                issues.append({"severity": "warning", "split": split, "message": f"Unexpected image size {img_path.name}: {(w, h)}"})
            try:
                rows = load_yolo_label(lab_path)
                validate_yolo_rows(rows, len(CLASS_TO_IDX), lab_path)
            except Exception as e:
                issues.append({"severity": "error", "split": split, "message": str(e)})
    issues_df = pd.DataFrame(issues)
    issues_df.to_csv(dataset_root / "validation_issues.csv", index=False)
    return issues_df


def draw_random_samples(dataset_root: Path, split: str = "train", n: int = 9, out_path: Optional[Path] = None):
    img_dir = dataset_root / "images" / split
    lab_dir = dataset_root / "labels" / split
    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        return
    rng = random.Random(CFG.SEED)
    chosen = rng.sample(image_paths, k=min(n, len(image_paths)))
    cols = 3
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(chosen):]:
        ax.axis("off")
    for ax, img_path in zip(axes, chosen):
        img = read_image_rgb(img_path)
        lab_path = lab_dir / f"{img_path.stem}.txt"
        labels = load_yolo_label(lab_path)
        for row in labels:
            box = yolo_to_xyxy(row["bbox_yolo"], img.shape[1], img.shape[0])
            x1, y1, x2, y2 = map(int, box)
            cls_name = IDX_TO_CLASS[row["class_id"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, cls_name, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        ax.imshow(img)
        ax.set_title(img_path.stem)
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180)
    plt.show()
    plt.close(fig)


ISSUES_DF = run_dataset_checks(MERGED_DATASET_ROOT)
print("Validation issues:", len(ISSUES_DF))
print(ISSUES_DF.head(20))
draw_random_samples(MERGED_DATASET_ROOT, "train", out_path=MERGED_DATASET_ROOT / "plots" / "train_samples.png")

# %% [markdown]
# ## 9. Shared Training Files

# %%
def write_common_hyp_files():
    common_hyp = {
        "lr0": 0.01,
        "lrf": 0.1,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 0.05,
        "cls": 0.3,
        "cls_pw": 1.0,
        "obj": 1.0,
        "obj_pw": 1.0,
        "iou_t": 0.20,
        "anchor_t": 4.0,
        "fl_gamma": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.50,
        "hsv_v": 0.30,
        "degrees": 0.0,
        "translate": 0.02,
        "scale": 0.05,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }
    path = Path(CFG.CONFIGS_DIR) / "hyp_uav_paper_style.yaml"
    save_yaml(common_hyp, path)
    return path


COMMON_HYP_PATH = write_common_hyp_files()
print(COMMON_HYP_PATH)

# %% [markdown]
# ## 10. Model-Specific Training Helpers

# %%
def effective_model_spec(model_name: str) -> ModelSpec:
    spec = MODEL_SPECS[model_name]
    spec = ModelSpec(**asdict(spec))
    if CFG.EPOCHS_OVERRIDE is not None:
        spec.epochs = CFG.EPOCHS_OVERRIDE
    if CFG.DRY_RUN or CFG.SANITY_MODE:
        spec.epochs = CFG.DRY_RUN_EPOCHS
        spec.batch_size = min(spec.batch_size, 4 if GPU_COUNT else 2)
    elif spec.model_name == "yolov8":
        tuned_img_size = recommended_yolov8_img_size(spec.img_size)
        if tuned_img_size != spec.img_size:
            print(
                f"Accuracy-tuned YOLOv8 image size from {spec.img_size} to {tuned_img_size} "
                f"for {PRIMARY_GPU_NAME} ({PRIMARY_GPU_MEMORY_GB:.1f} GB VRAM)."
            )
            spec.img_size = tuned_img_size
        tuned_batch_size = recommended_yolov8_batch_size(
            spec.batch_size,
            spec.img_size,
            accuracy_mode=CFG.YOLOV8_ACCURACY_MODE,
        )
        if tuned_batch_size != spec.batch_size:
            print(
                f"Auto-tuned YOLOv8 batch size from {spec.batch_size} to {tuned_batch_size} "
                f"for {PRIMARY_GPU_NAME} ({PRIMARY_GPU_MEMORY_GB:.1f} GB VRAM)."
            )
            spec.batch_size = tuned_batch_size
    return spec


def experiment_dir(model_name: str) -> Path:
    return Path(CFG.EXPERIMENTS_DIR) / model_name


def darknet_max_batches_for_spec(spec: ModelSpec, train_image_count: int) -> int:
    default_max_batches = max(6000, len(CFG.TARGET_CLASSES) * 2000)
    if not (CFG.EPOCHS_OVERRIDE is not None or CFG.DRY_RUN or CFG.SANITY_MODE):
        return default_max_batches
    steps_per_epoch = max(1, math.ceil(train_image_count / max(spec.batch_size, 1)))
    target_max_batches = max(1, spec.epochs) * steps_per_epoch
    print(
        f"Darknet epoch override active: train_images={train_image_count}, batch={spec.batch_size}, "
        f"epochs={spec.epochs}, steps_per_epoch={steps_per_epoch}, max_batches={target_max_batches}"
    )
    return max(steps_per_epoch, target_max_batches)


def maybe_download_file(url_or_name: str, dst_dir: Path, cwd: Optional[Path] = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if url_or_name.startswith("http://") or url_or_name.startswith("https://"):
        out_path = dst_dir / Path(url_or_name).name
        if out_path.exists():
            return out_path
        if not internet_available():
            raise RuntimeError(f"Internet access is required to download {url_or_name}")
        run_cmd(["wget", "-O", str(out_path), url_or_name], cwd=cwd, log_path=Path(CFG.LOGS_DIR) / f"download_{out_path.name}.log")
        return out_path
    return Path(url_or_name)


def prepare_darknet_custom_files(spec: ModelSpec) -> Dict[str, Path]:
    repo_dir = Path(CFG.REPOS_DIR) / spec.repo_name
    exp_dir = experiment_dir(spec.model_name)
    cfg_dir = exp_dir / "darknet_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    names_path = cfg_dir / "obj.names"
    with names_path.open("w", encoding="utf-8") as f:
        for name in CFG.TARGET_CLASSES:
            f.write(f"{name}\n")

    train_txt = cfg_dir / "train.txt"
    val_txt = cfg_dir / "val.txt"
    test_txt = cfg_dir / "test.txt"
    train_image_count = 0
    for split_name, path in [("train", train_txt), ("val", val_txt), ("test", test_txt)]:
        image_paths = sorted((MERGED_DATASET_ROOT / "images" / split_name).glob("*.jpg"))
        if split_name == "train":
            train_image_count = len(image_paths)
        with path.open("w", encoding="utf-8") as f:
            for img_path in image_paths:
                f.write(str(img_path) + "\n")

    data_path = cfg_dir / "obj.data"
    data_text = "\n".join(
        [
            f"classes = {len(CFG.TARGET_CLASSES)}",
            f"train = {train_txt}",
            f"valid = {val_txt}",
            f"names = {names_path}",
            f"backup = {exp_dir / 'backup'}",
        ]
    )
    data_path.write_text(data_text, encoding="utf-8")

    template_candidates = [
        repo_dir / "cfg" / "yolov4-tiny-custom.cfg",
        repo_dir / "cfg" / "yolov4-tiny.cfg",
    ]
    template_path = next((p for p in template_candidates if p.exists()), None)
    if template_path is None:
        raise FileNotFoundError("Could not find YOLOv4-tiny cfg template inside darknet repo.")

    max_batches = darknet_max_batches_for_spec(spec, train_image_count)
    steps = (int(max_batches * 0.8), int(max_batches * 0.9))
    filters = (len(CFG.TARGET_CLASSES) + 5) * 3
    lines = template_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("batch="):
            new_lines.append(f"batch={spec.batch_size}")
        elif stripped.startswith("subdivisions="):
            new_lines.append("subdivisions=16")
        elif stripped.startswith("width="):
            new_lines.append(f"width={CFG.IMG_SIZE}")
        elif stripped.startswith("height="):
            new_lines.append(f"height={CFG.IMG_SIZE}")
        elif stripped.startswith("max_batches"):
            new_lines.append(f"max_batches={max_batches}")
        elif stripped.startswith("steps="):
            new_lines.append(f"steps={steps[0]},{steps[1]}")
        else:
            new_lines.append(line)

    lines = new_lines
    yolo_indices = [i for i, line in enumerate(lines) if line.strip() == "[yolo]"]
    for yi in yolo_indices:
        j = yi + 1
        while j < len(lines) and not lines[j].startswith("["):
            if lines[j].strip().startswith("classes="):
                lines[j] = f"classes={len(CFG.TARGET_CLASSES)}"
            j += 1
        k = yi - 1
        while k >= 0:
            if lines[k].strip().startswith("filters="):
                lines[k] = f"filters={filters}"
                break
            k -= 1
    custom_cfg_path = cfg_dir / "yolov4_tiny_custom.cfg"
    custom_cfg_path.write_text("\n".join(lines), encoding="utf-8")

    pretrained_path = maybe_download_file(spec.pretrained_weights, cfg_dir)
    return {
        "repo_dir": repo_dir,
        "data_path": data_path,
        "names_path": names_path,
        "cfg_path": custom_cfg_path,
        "pretrained_path": pretrained_path,
        "train_txt": train_txt,
        "val_txt": val_txt,
        "test_txt": test_txt,
    }


def build_ddp_prefix(nproc: int) -> List[str]:
    port = 29500 + (CFG.SEED % 1000)
    return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc), "--master_port", str(port)]


def train_yolov4_darknet(spec: ModelSpec) -> Path:
    exp_dir = experiment_dir(spec.model_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    dark = prepare_darknet_custom_files(spec)
    repo_dir = dark["repo_dir"]
    backup_dir = exp_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = backup_dir / "yolov4_tiny_custom_last.weights"
    best_ckpt = backup_dir / "yolov4_tiny_custom_best.weights"

    darknet_bin = repo_dir / "darknet"
    weights_in = last_ckpt if (CFG.RESUME and last_ckpt.exists()) else dark["pretrained_path"]
    cmd = [
        str(darknet_bin),
        "detector",
        "train",
        str(dark["data_path"]),
        str(dark["cfg_path"]),
        str(weights_in),
        "-dont_show",
        "-map",
    ]
    if GPU_COUNT > 1:
        cmd += ["-gpus", ",".join(map(str, range(GPU_COUNT)))]
    run_cmd(cmd, cwd=repo_dir, log_path=exp_dir / "train.log")
    if best_ckpt.exists():
        return best_ckpt
    candidates = sorted(backup_dir.glob("*.weights"))
    if not candidates:
        raise FileNotFoundError("YOLOv4 training finished but no weights were found in the backup directory.")
    return candidates[-1]


def train_yolov5_family(spec: ModelSpec) -> Path:
    repo_dir = Path(CFG.REPOS_DIR) / spec.repo_name
    exp_dir = experiment_dir(spec.model_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{spec.model_name}_run"
    run_root = exp_dir / "train"
    resume_ckpt = run_root / run_name / "weights" / "last.pt"

    if CFG.RESUME and resume_ckpt.exists():
        cmd = [sys.executable, spec.train_script, "--resume", str(resume_ckpt)]
        run_cmd(cmd, cwd=repo_dir, log_path=exp_dir / "train.log")
        return run_root / run_name / "weights" / "best.pt"

    base_cmd = [
        spec.train_script,
        "--img",
        str(spec.img_size),
        "--batch",
        str(spec.batch_size),
        "--epochs",
        str(spec.epochs),
        "--data",
        str(MERGED_DATASET_ROOT / "data.yaml"),
        "--weights",
        spec.pretrained_weights,
        "--project",
        str(run_root),
        "--name",
        run_name,
        "--workers",
        str(CFG.WORKERS),
        "--patience",
        str(CFG.PATIENCE),
        "--cache",
        "disk",
        "--exist-ok",
        "--seed",
        str(CFG.SEED),
        "--hyp",
        str(COMMON_HYP_PATH if spec.model_name == "yolov5" else repo_dir / spec.hyp_file),
    ]
    if spec.cfg:
        base_cmd += ["--cfg", spec.cfg]
    if spec.optimizer.lower() == "adam":
        base_cmd += ["--adam"]
    nproc = GPU_COUNT if GPU_COUNT > 1 else 1
    cmd = build_ddp_prefix(nproc) + base_cmd if GPU_COUNT > 1 else [sys.executable] + base_cmd
    run_cmd(cmd, cwd=repo_dir, log_path=exp_dir / "train.log")
    best_pt = run_root / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but {best_pt} was not found.")
    return best_pt


def patch_yolov7_custom_cfg(repo_dir: Path, cfg_rel: str, nc: int) -> Path:
    src = repo_dir / cfg_rel
    if src.exists():
        text = src.read_text(encoding="utf-8")
    else:
        base = repo_dir / "cfg" / "training" / "yolov7-w6.yaml"
        text = base.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        if line.strip().startswith("nc:"):
            out_lines.append(f"nc: {nc}")
        else:
            out_lines.append(line)
    out_path = Path(CFG.CONFIGS_DIR) / "yolov7_w6_custom_generated.yaml"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    return out_path


def train_yolov7(spec: ModelSpec) -> Path:
    repo_dir = Path(CFG.REPOS_DIR) / spec.repo_name
    exp_dir = experiment_dir(spec.model_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{spec.model_name}_run"
    run_root = exp_dir / "train"
    resume_ckpt = run_root / run_name / "weights" / "last.pt"
    if CFG.RESUME and resume_ckpt.exists():
        cmd = [sys.executable, spec.train_script, "--resume", str(resume_ckpt)]
        run_cmd(cmd, cwd=repo_dir, log_path=exp_dir / "train.log")
        return run_root / run_name / "weights" / "best.pt"

    cfg_path = patch_yolov7_custom_cfg(repo_dir, spec.cfg, len(CFG.TARGET_CLASSES))
    base_cmd = [
        spec.train_script,
        "--workers",
        str(CFG.WORKERS),
        "--batch-size",
        str(spec.batch_size),
        "--data",
        str(MERGED_DATASET_ROOT / "data.yaml"),
        "--img",
        str(spec.img_size),
        str(spec.img_size),
        "--cfg",
        str(cfg_path),
        "--weights",
        spec.pretrained_weights,
        "--name",
        run_name,
        "--project",
        str(run_root),
        "--hyp",
        str(COMMON_HYP_PATH),
    ]
    if GPU_COUNT > 1:
        base_cmd += ["--device", DEVICE_STRING, "--sync-bn"]
    nproc = GPU_COUNT if GPU_COUNT > 1 else 1
    cmd = build_ddp_prefix(nproc) + base_cmd if GPU_COUNT > 1 else [sys.executable] + base_cmd
    run_cmd(cmd, cwd=repo_dir, log_path=exp_dir / "train.log")
    best_pt = run_root / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but {best_pt} was not found.")
    return best_pt


def _is_valid_torch_checkpoint(path: Path) -> bool:
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except TypeError:
        try:
            torch.load(path, map_location="cpu")
            return True
        except Exception as e:
            print(f"Invalid checkpoint {path}: {e}")
            return False
    except Exception as e:
        print(f"Invalid checkpoint {path}: {e}")
        return False


def _purge_corrupt_ultralytics_weight_alias(weight_name: Optional[str]):
    if not weight_name:
        return
    weight_path = Path(weight_name)
    if weight_path.suffix != ".pt":
        return
    candidates = [weight_path]
    if not weight_path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / weight_path.name,
                KAGGLE_WORKING / weight_path.name if IS_KAGGLE else Path.cwd() / "working" / weight_path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and not _is_valid_torch_checkpoint(candidate):
            with contextlib.suppress(Exception):
                candidate.unlink()
                print(f"Deleted corrupt checkpoint alias: {candidate}")


def train_yolov8(spec: ModelSpec) -> Path:
    from ultralytics import YOLO

    exp_dir = experiment_dir(spec.model_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{spec.model_name}_run"
    project_dir = exp_dir / "train"
    save_dir = project_dir / run_name
    weights_last = save_dir / "weights" / "last.pt"
    weights_best = save_dir / "weights" / "best.pt"

    _purge_corrupt_ultralytics_weight_alias(spec.pretrained_weights)
    resume_from_last = CFG.RESUME and weights_last.exists() and _is_valid_torch_checkpoint(weights_last)
    workers = resolved_worker_count(CFG.WORKERS)
    batch_candidates = [spec.batch_size] if resume_from_last else yolov8_batch_candidates(spec.batch_size)
    print(
        f"YOLOv8 training profile: gpu={PRIMARY_GPU_NAME}, vram_gb={PRIMARY_GPU_MEMORY_GB:.1f}, "
        f"workers={workers}, batch_candidates={batch_candidates}, resume={resume_from_last}"
    )

    results = None
    last_error = None
    for batch_size in batch_candidates:
        model = YOLO(str(weights_last)) if resume_from_last else YOLO(spec.pretrained_weights)
        train_args = dict(
            data=str(MERGED_DATASET_ROOT / "data.yaml"),
            imgsz=spec.img_size,
            epochs=spec.epochs,
            batch=batch_size,
            workers=workers,
            patience=CFG.PATIENCE,
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            device=DEVICE_STRING if torch.cuda.is_available() else "cpu",
            seed=CFG.SEED,
            cache="disk",
            amp=bool(torch.cuda.is_available()),
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.25,
            degrees=3.0 if CFG.YOLOV8_ACCURACY_MODE else 0.0,
            translate=0.03 if CFG.YOLOV8_ACCURACY_MODE else 0.02,
            scale=0.10 if CFG.YOLOV8_ACCURACY_MODE else 0.05,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5 if CFG.YOLOV8_ACCURACY_MODE else 0.0,
            mosaic=0.20 if CFG.YOLOV8_ACCURACY_MODE else 0.0,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=10 if CFG.YOLOV8_ACCURACY_MODE else 0,
            cos_lr=True,
            resume=bool(resume_from_last),
            plots=False,
            verbose=True,
        )
        try:
            print(f"Starting YOLOv8 training with batch={batch_size}, workers={workers}")
            results = model.train(**train_args)
            break
        except AttributeError as e:
            if "curves_results" in str(e) and (weights_best.exists() or weights_last.exists()):
                print("Ignoring known Ultralytics curves_results bug and continuing with saved weights.")
                break
            raise
        except RuntimeError as e:
            oom = "out of memory" in str(e).lower()
            if not oom or batch_size == batch_candidates[-1]:
                raise
            last_error = e
            print(f"CUDA OOM at batch={batch_size}; retrying with a smaller batch.")
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            continue

    if results is None and last_error is not None:
        raise last_error

    if results is not None and hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)
        weights_last = save_dir / "weights" / "last.pt"
        weights_best = save_dir / "weights" / "best.pt"

    for ckpt in [weights_best, weights_last]:
        if ckpt.exists() and _is_valid_torch_checkpoint(ckpt):
            return ckpt
    raise FileNotFoundError(f"No valid YOLOv8 checkpoint found in {save_dir / 'weights'}")


def train_one_model(model_name: str) -> Path:
    spec = effective_model_spec(model_name)
    if spec.family == "darknet":
        return train_yolov4_darknet(spec)
    if spec.family in {"yolov5", "tph"}:
        return train_yolov5_family(spec)
    if spec.family == "yolov7":
        return train_yolov7(spec)
    if spec.family == "ultralytics":
        return train_yolov8(spec)
    raise ValueError(f"Unsupported family: {spec.family}")

# %% [markdown]
# ## 11. Prediction Export Helpers

# %%
def export_predictions_yolov4(weights_path: Path, split: str, out_dir: Path) -> Path:
    spec = MODEL_SPECS["yolov4"]
    dark = prepare_darknet_custom_files(spec)
    labels_out = out_dir / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)
    net = cv2.dnn.readNetFromDarknet(str(dark["cfg_path"]), str(weights_path))
    if GPU_COUNT:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    infer_img_size = effective_model_spec("yolov4").img_size
    model.setInputParams(scale=1 / 255.0, size=(infer_img_size, infer_img_size), swapRB=True)
    image_dir = MERGED_DATASET_ROOT / "images" / split
    for img_path in tqdm(sorted(image_dir.glob("*.jpg")), desc=f"YOLOv4 predict {split}"):
        image = cv2.imread(str(img_path))
        class_ids, scores, boxes = model.detect(image, confThreshold=CFG.CONF_SWEEP_START, nmsThreshold=0.6)
        out_txt = labels_out / f"{img_path.stem}.txt"
        with out_txt.open("w", encoding="utf-8") as f:
            if len(class_ids) == 0:
                continue
            for cls_id, score, box in zip(class_ids.flatten(), scores.flatten(), boxes):
                x, y, w, h = map(float, box)
                yolo_box = xyxy_to_yolo([x, y, x + w, y + h], image.shape[1], image.shape[0])
                f.write(f"{int(cls_id)} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {float(score):.6f}\n")
    return labels_out


def export_predictions_with_cli(model_name: str, weights_path: Path, split: str, out_dir: Path) -> Path:
    spec = MODEL_SPECS[model_name]
    repo_dir = Path(CFG.REPOS_DIR) / spec.repo_name
    source_dir = MERGED_DATASET_ROOT / "images" / split
    infer_img_size = effective_model_spec(model_name).img_size
    cmd = [
        sys.executable,
        spec.detect_script,
        "--weights",
        str(weights_path),
        "--img",
        str(infer_img_size),
        "--conf",
        str(CFG.CONF_SWEEP_START),
        "--source",
        str(source_dir),
        "--save-txt",
        "--save-conf",
        "--exist-ok",
        "--project",
        str(out_dir),
        "--name",
        split,
    ]
    if model_name == "yolov7":
        cmd = [
            sys.executable,
            spec.detect_script,
            "--weights",
            str(weights_path),
            "--img-size",
            str(infer_img_size),
            "--conf",
            str(CFG.CONF_SWEEP_START),
            "--source",
            str(source_dir),
            "--save-txt",
            "--save-conf",
            "--nosave",
            "--project",
            str(out_dir),
            "--name",
            split,
            "--exist-ok",
        ]
    run_cmd(cmd, cwd=repo_dir, log_path=out_dir / f"{split}_predict.log")
    pred_labels = out_dir / split / "labels"
    if not pred_labels.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_labels}")
    return pred_labels


def export_predictions_yolov8(weights_path: Path, split: str, out_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    infer_img_size = effective_model_spec("yolov8").img_size
    model.predict(
        source=str(MERGED_DATASET_ROOT / "images" / split),
        conf=CFG.CONF_SWEEP_START,
        iou=0.6,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(out_dir),
        name=split,
        exist_ok=True,
        imgsz=infer_img_size,
        augment=CFG.YOLOV8_EVAL_TTA,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    pred_labels = out_dir / split / "labels"
    if not pred_labels.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_labels}")
    return pred_labels


def export_predictions(model_name: str, weights_path: Path, split: str) -> Path:
    out_dir = experiment_dir(model_name) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "yolov4":
        return export_predictions_yolov4(weights_path, split, out_dir)
    if model_name in {"yolov5", "yolov5_tph", "yolov7"}:
        return export_predictions_with_cli(model_name, weights_path, split, out_dir)
    if model_name == "yolov8":
        return export_predictions_yolov8(weights_path, split, out_dir)
    raise ValueError(model_name)

# %% [markdown]
# ## 12. Common Evaluation

# %%
def collect_gt_and_preds(split: str, pred_label_dir: Path):
    gt = {}
    pred = {}
    image_dir = MERGED_DATASET_ROOT / "images" / split
    label_dir = MERGED_DATASET_ROOT / "labels" / split
    for img_path in sorted(image_dir.glob("*.jpg")):
        lab_path = label_dir / f"{img_path.stem}.txt"
        gt[img_path.stem] = load_yolo_label(lab_path)
    for txt in sorted(pred_label_dir.glob("*.txt")):
        pred[txt.stem] = load_yolo_label(txt, with_conf=True)
    return gt, pred


def compute_ap50(gt_by_image: Dict[str, List[dict]], pred_by_image: Dict[str, List[dict]], num_classes: int, iou_thr: float = 0.5, split_name: str = "test"):
    ap_per_class = {}
    pr_curves = {}
    for class_id in range(num_classes):
        gt_class = {}
        total_gt = 0
        for image_id, rows in gt_by_image.items():
            boxes = [row["bbox_yolo"] for row in rows if row["class_id"] == class_id]
            gt_class[image_id] = {"boxes": boxes, "matched": np.zeros(len(boxes), dtype=bool)}
            total_gt += len(boxes)
        preds = []
        for image_id, rows in pred_by_image.items():
            for row in rows:
                if row["class_id"] == class_id:
                    preds.append((image_id, row["conf"], row["bbox_yolo"]))
        preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(preds), dtype=float)
        fp = np.zeros(len(preds), dtype=float)
        for i, (image_id, conf, box) in enumerate(preds):
            gt_boxes = gt_class.get(image_id, {"boxes": [], "matched": np.array([], dtype=bool)})
            if not gt_boxes["boxes"]:
                fp[i] = 1
                continue
            image_path = MERGED_DATASET_ROOT / "images" / split_name / f"{image_id}.jpg"
            with Image.open(image_path) as im:
                w, h = im.size
            pred_xyxy = yolo_to_xyxy(box, w, h)
            ious = [box_iou(pred_xyxy, yolo_to_xyxy(gt_box, w, h)) for gt_box in gt_boxes["boxes"]]
            if not ious:
                fp[i] = 1
                continue
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]
            if best_iou >= iou_thr and not gt_boxes["matched"][best_idx]:
                tp[i] = 1
                gt_boxes["matched"][best_idx] = True
            else:
                fp[i] = 1
        if total_gt == 0:
            ap_per_class[IDX_TO_CLASS[class_id]] = float("nan")
            pr_curves[IDX_TO_CLASS[class_id]] = {"precision": [], "recall": [], "score": []}
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / total_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        recall_points = np.linspace(0, 1, 101)
        precision_interp = np.zeros_like(recall_points)
        for j, r in enumerate(recall_points):
            precision_interp[j] = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap = precision_interp.mean()
        ap_per_class[IDX_TO_CLASS[class_id]] = float(ap)
        pr_curves[IDX_TO_CLASS[class_id]] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "score": [p[1] for p in preds],
        }
    map50 = float(np.nanmean(list(ap_per_class.values())))
    return map50, ap_per_class, pr_curves


def confusion_matrix_at_threshold(gt_by_image: Dict[str, List[dict]], pred_by_image: Dict[str, List[dict]], conf_thr: float, split_name: str, iou_thr: float = 0.5):
    nc = len(CFG.TARGET_CLASSES)
    bg = nc
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for image_id, gt_rows in gt_by_image.items():
        preds = [p for p in pred_by_image.get(image_id, []) if p["conf"] >= conf_thr]
        image_path = MERGED_DATASET_ROOT / "images" / split_name / f"{image_id}.jpg"
        if not image_path.exists():
            continue
        with Image.open(image_path) as im:
            w, h = im.size

        gt_xyxy = [yolo_to_xyxy(row["bbox_yolo"], w, h) for row in gt_rows]
        pred_xyxy = [yolo_to_xyxy(row["bbox_yolo"], w, h) for row in preds]

        candidate_matches = []
        for pi, pbox in enumerate(pred_xyxy):
            for gi, gbox in enumerate(gt_xyxy):
                iou = box_iou(pbox, gbox)
                if iou >= iou_thr:
                    candidate_matches.append((iou, pi, gi))
        candidate_matches.sort(reverse=True)

        used_pred = set()
        used_gt = set()
        matches = []
        for iou, pi, gi in candidate_matches:
            if pi in used_pred or gi in used_gt:
                continue
            used_pred.add(pi)
            used_gt.add(gi)
            matches.append((pi, gi))

        for pi, gi in matches:
            cm[preds[pi]["class_id"], gt_rows[gi]["class_id"]] += 1
        for gi, gt_row in enumerate(gt_rows):
            if gi not in used_gt:
                cm[bg, gt_row["class_id"]] += 1
        for pi, pred_row in enumerate(preds):
            if pi not in used_pred:
                cm[pred_row["class_id"], bg] += 1
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> pd.DataFrame:
    rows = []
    for ci, cls_name in enumerate(CFG.TARGET_CLASSES):
        tp = cm[ci, ci]
        fp = cm[ci, :].sum() - tp
        fn = cm[:, ci].sum() - tp
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append(
            {
                "class_name": cls_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def confidence_sweep(gt_by_image, pred_by_image, split_name: str):
    thresholds = np.linspace(CFG.CONF_SWEEP_START, CFG.CONF_SWEEP_STOP, CFG.CONF_SWEEP_STEPS)
    rows = []
    for thr in thresholds:
        cm = confusion_matrix_at_threshold(gt_by_image, pred_by_image, float(thr), split_name, CFG.IOU_EVAL)
        metrics_df = metrics_from_confusion_matrix(cm)
        tp = int(sum(cm[i, i] for i in range(len(CFG.TARGET_CLASSES))))
        fp = int(cm[: len(CFG.TARGET_CLASSES), :].sum() - tp)
        fn = int(cm[:, : len(CFG.TARGET_CLASSES)].sum() - tp)
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append(
            {
                "confidence": float(thr),
                "precision_micro": precision,
                "recall_micro": recall,
                "f1_micro": f1,
                "precision_macro": float(metrics_df["precision"].mean()),
                "recall_macro": float(metrics_df["recall"].mean()),
                "f1_macro": float(metrics_df["f1"].mean()),
            }
        )
    sweep_df = pd.DataFrame(rows)
    best_row = sweep_df.sort_values(["f1_micro", "confidence"], ascending=[False, True]).iloc[0].to_dict()
    return sweep_df, best_row


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    labels = CFG.TARGET_CLASSES + ["background"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pr_curves(pr_curves: Dict[str, dict], out_path: Path):
    plt.figure(figsize=(8, 6))
    for cls_name, curve in pr_curves.items():
        if not curve["precision"]:
            continue
        plt.plot(curve["recall"], curve["precision"], label=cls_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_class_metric_bars(metrics_df: pd.DataFrame, ap_per_class: Dict[str, float], out_path: Path):
    plot_df = metrics_df.copy()
    plot_df["ap50"] = plot_df["class_name"].map(ap_per_class)
    melt = plot_df.melt(id_vars="class_name", value_vars=["precision", "recall", "f1", "ap50"], var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melt, x="class_name", y="value", hue="metric")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confidence_sweep(sweep_df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_df["confidence"], sweep_df["precision_micro"], label="Precision")
    plt.plot(sweep_df["confidence"], sweep_df["recall_micro"], label="Recall")
    plt.plot(sweep_df["confidence"], sweep_df["f1_micro"], label="F1")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score")
    plt.title("Confidence Sweep")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def count_parameters_from_checkpoint(weights_path: Path) -> Optional[int]:
    if weights_path.suffix != ".pt":
        return None
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")
    except Exception:
        return None
    if isinstance(ckpt, dict):
        if "model" in ckpt and hasattr(ckpt["model"], "parameters"):
            return int(sum(p.numel() for p in ckpt["model"].parameters()))
        if "state_dict" in ckpt:
            return int(sum(v.numel() for v in ckpt["state_dict"].values() if hasattr(v, "numel")))
    if hasattr(ckpt, "parameters"):
        return int(sum(p.numel() for p in ckpt.parameters()))
    return None


def approximate_inference_speed(model_name: str, weights_path: Path, split: str = "test") -> float:
    image_paths = sorted((MERGED_DATASET_ROOT / "images" / split).glob("*.jpg"))[: CFG.SPEED_EVAL_IMAGES]
    if not image_paths:
        return float("nan")
    infer_img_size = effective_model_spec(model_name).img_size
    if model_name == "yolov4":
        spec = MODEL_SPECS["yolov4"]
        dark = prepare_darknet_custom_files(spec)
        net = cv2.dnn.readNetFromDarknet(str(dark["cfg_path"]), str(weights_path))
        if GPU_COUNT:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255.0, size=(infer_img_size, infer_img_size), swapRB=True)
        _ = model.detect(cv2.imread(str(image_paths[0])), confThreshold=0.25, nmsThreshold=0.6)
        t0 = time.perf_counter()
        for img_path in image_paths:
            _ = model.detect(cv2.imread(str(img_path)), confThreshold=0.25, nmsThreshold=0.6)
        dt = time.perf_counter() - t0
        return 1000.0 * dt / len(image_paths)
    if model_name == "yolov8":
        from ultralytics import YOLO

        model = YOLO(str(weights_path))
        _ = model.predict(source=str(image_paths[0]), imgsz=infer_img_size, conf=0.25, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        for img_path in image_paths:
            _ = model.predict(source=str(img_path), imgsz=infer_img_size, conf=0.25, verbose=False, device=0 if torch.cuda.is_available() else "cpu")
        dt = time.perf_counter() - t0
        return 1000.0 * dt / len(image_paths)
    return float("nan")


def evaluate_model(model_name: str, weights_path: Path) -> Dict[str, object]:
    exp_dir = experiment_dir(model_name)
    metrics_dir = exp_dir / "metrics"
    plots_dir = exp_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    pred_val_dir = export_predictions(model_name, weights_path, "val")
    pred_test_dir = export_predictions(model_name, weights_path, "test")

    gt_val, pred_val = collect_gt_and_preds("val", pred_val_dir)
    gt_test, pred_test = collect_gt_and_preds("test", pred_test_dir)

    val_map50, val_ap_per_class, _ = compute_ap50(gt_val, pred_val, len(CFG.TARGET_CLASSES), CFG.IOU_EVAL, "val")
    test_map50, test_ap_per_class, test_pr_curves = compute_ap50(gt_test, pred_test, len(CFG.TARGET_CLASSES), CFG.IOU_EVAL, "test")

    sweep_df, best_conf_row = confidence_sweep(gt_test, pred_test, "test")
    best_conf = float(best_conf_row["confidence"])
    standard_conf = float(CFG.CONFIDENCE_STANDARD)

    cm_best = confusion_matrix_at_threshold(gt_test, pred_test, best_conf, "test", CFG.IOU_EVAL)
    metrics_best = metrics_from_confusion_matrix(cm_best)
    cm_standard = confusion_matrix_at_threshold(gt_test, pred_test, standard_conf, "test", CFG.IOU_EVAL)
    metrics_standard = metrics_from_confusion_matrix(cm_standard)

    plot_confusion_matrix(cm_best, f"{model_name} confusion matrix @ best F1 conf={best_conf:.2f}", plots_dir / "confusion_matrix_best.png")
    plot_confusion_matrix(cm_standard, f"{model_name} confusion matrix @ conf={standard_conf:.2f}", plots_dir / "confusion_matrix_025.png")
    plot_pr_curves(test_pr_curves, plots_dir / "pr_curves_test.png")
    plot_class_metric_bars(metrics_best, test_ap_per_class, plots_dir / "class_metrics_best.png")
    plot_confidence_sweep(sweep_df, plots_dir / "confidence_sweep.png")

    metrics_best.to_csv(metrics_dir / "per_class_metrics_best.csv", index=False)
    metrics_standard.to_csv(metrics_dir / "per_class_metrics_025.csv", index=False)
    sweep_df.to_csv(metrics_dir / "confidence_sweep.csv", index=False)
    save_json(val_ap_per_class, metrics_dir / "val_ap50_per_class.json")
    save_json(test_ap_per_class, metrics_dir / "test_ap50_per_class.json")

    tp_best = int(sum(cm_best[i, i] for i in range(len(CFG.TARGET_CLASSES))))
    fp_best = int(cm_best[: len(CFG.TARGET_CLASSES), :].sum() - tp_best)
    fn_best = int(cm_best[:, : len(CFG.TARGET_CLASSES)].sum() - tp_best)
    best_precision = tp_best / max(tp_best + fp_best, 1e-9)
    best_recall = tp_best / max(tp_best + fn_best, 1e-9)
    best_f1 = 2 * best_precision * best_recall / max(best_precision + best_recall, 1e-9)

    tp_std = int(sum(cm_standard[i, i] for i in range(len(CFG.TARGET_CLASSES))))
    fp_std = int(cm_standard[: len(CFG.TARGET_CLASSES), :].sum() - tp_std)
    fn_std = int(cm_standard[:, : len(CFG.TARGET_CLASSES)].sum() - tp_std)
    std_precision = tp_std / max(tp_std + fp_std, 1e-9)
    std_recall = tp_std / max(tp_std + fn_std, 1e-9)
    std_f1 = 2 * std_precision * std_recall / max(std_precision + std_recall, 1e-9)

    params = count_parameters_from_checkpoint(weights_path)
    speed_ms = approximate_inference_speed(model_name, weights_path)

    results = {
        "model_name": model_name,
        "display_name": MODEL_SPECS[model_name].display_name,
        "weights_path": str(weights_path),
        "best_val_map50": val_map50,
        "test_map50": test_map50,
        "best_confidence": best_conf,
        "test_precision_best_conf": best_precision,
        "test_recall_best_conf": best_recall,
        "test_f1_best_conf": best_f1,
        "test_precision_conf_025": std_precision,
        "test_recall_conf_025": std_recall,
        "test_f1_conf_025": std_f1,
        "params": params,
        "inference_ms_per_image": speed_ms,
        "paper_reference_map50": CFG.PAPER_REFERENCE_MAP50.get(model_name, float("nan")),
        "notes": MODEL_SPECS[model_name].notes,
    }
    save_json(results, metrics_dir / "summary.json")
    return results

# %% [markdown]
# ## 13. Training And Evaluation Driver

# %%
def train_and_evaluate_model(model_name: str) -> Dict[str, object]:
    spec = effective_model_spec(model_name)
    exp_dir = experiment_dir(model_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    try:
        weights_path = train_one_model(model_name)
        result = evaluate_model(model_name, Path(weights_path))
        result["status"] = "success"
        return result
    except Exception as e:
        failure = {
            "model_name": model_name,
            "display_name": spec.display_name,
            "status": "failed",
            "best_val_map50": float("nan"),
            "test_map50": float("nan"),
            "best_confidence": float("nan"),
            "test_precision_best_conf": float("nan"),
            "test_recall_best_conf": float("nan"),
            "test_f1_best_conf": float("nan"),
            "test_precision_conf_025": float("nan"),
            "test_recall_conf_025": float("nan"),
            "test_f1_conf_025": float("nan"),
            "params": None,
            "inference_ms_per_image": float("nan"),
            "paper_reference_map50": CFG.PAPER_REFERENCE_MAP50.get(model_name, float("nan")),
            "notes": f"{spec.notes} Failure: {repr(e)}",
        }
        save_json(failure, exp_dir / "failure.json")
        print(f"Model {model_name} failed:", repr(e))
        return failure


TRAIN_SEQUENCE = CFG.MODEL_SEQUENCE if CFG.TRAIN_MODEL_SEQUENCE else [CFG.MODEL_NAME]
COMPARISON_ROWS = []
for model_name in TRAIN_SEQUENCE:
    COMPARISON_ROWS.append(train_and_evaluate_model(model_name))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

COMPARISON_DF = pd.DataFrame(COMPARISON_ROWS)
COMPARISON_DF.to_csv(Path(CFG.SUMMARIES_DIR) / "model_comparison.csv", index=False)
print(COMPARISON_DF)

# %% [markdown]
# ## 14. Final Comparison Plots

# %%
def plot_model_comparison(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    keep = plot_df["status"] == "success"
    if not keep.any():
        return
    plot_df = plot_df[keep].copy()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="display_name", y="best_val_map50")
    plt.xticks(rotation=20)
    plt.ylabel("Validation mAP@0.5")
    plt.title("Best validation mAP@0.5 by model")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_map50.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="display_name", y="test_f1_best_conf")
    plt.xticks(rotation=20)
    plt.ylabel("Test F1 at best confidence")
    plt.title("Test F1 by model")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_test_f1.png", dpi=180)
    plt.close()

    if plot_df["inference_ms_per_image"].notna().any():
        plt.figure(figsize=(10, 5))
        sns.barplot(data=plot_df, x="display_name", y="inference_ms_per_image")
        plt.xticks(rotation=20)
        plt.ylabel("ms / image")
        plt.title("Approximate inference speed")
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_speed.png", dpi=180)
        plt.close()


plot_model_comparison(COMPARISON_DF, Path(CFG.SUMMARIES_DIR) / "comparison_plots")

# %% [markdown]
# ## 15. Artifact Summary

# %%
artifact_summary = {
    "root_out": CFG.ROOT_OUT,
    "merged_dataset_root": str(MERGED_DATASET_ROOT),
    "spain_dataset_root": str(SPAIN_DATASET_ROOT),
    "china_dataset_root": str(CHINA_DATASET_ROOT),
    "summaries_dir": CFG.SUMMARIES_DIR,
    "experiments_dir": CFG.EXPERIMENTS_DIR,
    "comparison_csv": str(Path(CFG.SUMMARIES_DIR) / "model_comparison.csv"),
    "assumptions_json": str(Path(CFG.SUMMARIES_DIR) / "assumptions.json"),
    "dataset_yaml": str(MERGED_DATASET_ROOT / "data.yaml"),
}
save_json(artifact_summary, Path(CFG.SUMMARIES_DIR) / "artifact_summary.json")
print(json.dumps(artifact_summary, indent=2))
