"""CatBoost-классификатор пар + калибровка двух порогов.

API:
    train_model(X_train, y_train, cat_features, X_val=None, y_val=None) -> CatBoostClassifier
    predict_proba(model, X) -> np.ndarray
    save_model(model, path) / load_model(path) -> CatBoostClassifier
    calibrate_thresholds(y_true, prob, target_precision=0.99) -> dict
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve

from .config import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLDS_PATH


def _prepare(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """Категориальные признаки приводим к строкам — CatBoost ждёт object/string."""
    X = X.copy()
    for c in cat_features:
        X[c] = X[c].astype(int).astype(str)
    return X


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    iterations: int = 500,
    learning_rate: float = 0.08,
    depth: int = 6,
    random_seed: int = 42,
    verbose: int = 0,
) -> CatBoostClassifier:
    X_train = _prepare(X_train, cat_features)
    eval_set = None
    if X_val is not None and y_val is not None:
        X_val = _prepare(X_val, cat_features)
        eval_set = (X_val, y_val)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        cat_features=cat_features,
        task_type="CPU",
        verbose=verbose,
        early_stopping_rounds=50,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, eval_set=eval_set)
    return model


def predict_proba(model: CatBoostClassifier, X: pd.DataFrame, cat_features: list[str] | None = None) -> np.ndarray:
    """Возвращает P(match) для каждой пары."""
    if cat_features is None:
        cat_features = [c for c in X.columns if c.startswith("match_")]
    X = _prepare(X, cat_features)
    return model.predict_proba(X)[:, 1]


def save_model(model: CatBoostClassifier, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def calibrate_thresholds(
    y_true: np.ndarray,
    prob: np.ndarray,
    target_precision_auto: float = 0.99,
) -> dict:
    """Подбирает два порога: AUTO (precision>=target) и REVIEW (best F1)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    # AUTO: минимальный порог, при котором precision >= target
    auto_idx = np.where(prec[:-1] >= target_precision_auto)[0]
    auto_t = float(thr[auto_idx[0]]) if len(auto_idx) else 0.99

    # REVIEW: лучший F1
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    review_t = float(thr[best_idx])

    return {
        "auto_merge": auto_t,
        "review": review_t,
        "auto_precision": float(prec[auto_idx[0]]) if len(auto_idx) else 1.0,
        "review_precision": float(prec[best_idx]),
        "review_recall": float(rec[best_idx]),
        "review_f1": float(f1[best_idx]),
    }


def save_thresholds(thresholds: dict, path: Path | str = DEFAULT_THRESHOLDS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    return path


def load_thresholds(path: Path | str = DEFAULT_THRESHOLDS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
