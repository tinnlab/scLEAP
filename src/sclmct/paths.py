from __future__ import annotations
import os
from pathlib import Path

# repo root is two levels up from this file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = Path(os.environ.get("DATA_DIR",    PROJECT_ROOT / "data")).resolve()
MODELS_DIR  = Path(os.environ.get("MODELS_DIR",  PROJECT_ROOT / "models")).resolve()
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_ROOT / "results")).resolve()
CONFIGS_DIR = Path(os.environ.get("CONFIGS_DIR", PROJECT_ROOT / "configs")).resolve()

def ensure_dirs():
    for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
