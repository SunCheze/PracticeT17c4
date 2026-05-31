"""Конфигурация: пути, пороги, имена колонок.

Изменяй только тут — модули его импортируют.
"""
from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODELS_DIR / "catboost_er.cbm"
DEFAULT_THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

# ---- Колонки с поведенческими site-id (frozensets после collapse) ----
SITE_SET_COLS = [
    "fs_visited_30", "fs_visited_365", "fs_source_site_365", "fs_has_account",
    "fs_has_click_365", "fs_has_accept_365", "fs_has_order_30", "fs_has_order_365",
    "fs_source_site_30", "fs_has_click_30", "fs_has_accept_30", "fs_has_view_90",
]

# ---- Пороги решения (стартовые; калибруются на ваших данных) ----
THRESHOLD_AUTO_MERGE = 0.998   # auto-merge (precision >= 0.99 на val)
THRESHOLD_REVIEW = 0.95        # отправить в очередь оператору

# ---- Параметры блокинга ----
BLOCK_MAX_SIZE = 300           # большие блоки down-sample'ятся
SITE_STOP_MIN_DF = 150         # site-id чаще этого считается стоп-словом
