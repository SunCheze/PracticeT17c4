"""FastAPI-сервинг модели Entity Resolution.

Загружает модель из MLflow Model Registry по stage (Production или Staging) на
старте. Эндпоинты:
  - POST /score   — батч из пар профилей -> вероятности и рекомендации
  - GET /health   — kubernetes-style readiness/liveness
  - GET /metrics  — Prometheus метрики
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("MODEL_NAME", "entity-resolution")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production").capitalize()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
THRESHOLDS_PATH = Path(os.getenv("THRESHOLDS_PATH", "/app/models/thresholds.json"))

mlflow.set_tracking_uri(TRACKING_URI)

# --------------------------------------------------------------------------- #
# Prometheus метрики
# --------------------------------------------------------------------------- #
REQUESTS = Counter("er_requests_total", "Всего запросов", ["status", "stage"])
LATENCY = Histogram("er_latency_seconds", "Латентность инференса",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
PAIRS_SCORED = Counter("er_pairs_scored_total", "Сколько пар обработано")
ACTIONS = Counter("er_actions_total", "Рекомендации модели", ["action"])
MODEL_VERSION_GAUGE = Gauge("er_model_version", "Текущая версия модели", ["stage"])
PROB_MEAN = Gauge("er_prob_match_mean", "Средняя P(match) за последний запрос")

# --------------------------------------------------------------------------- #
# Загрузка модели
# --------------------------------------------------------------------------- #
app = FastAPI(title="Entity Resolution API")

_model: Any = None
_model_version: str = "unknown"
_thresholds: dict = {}


@app.on_event("startup")
def load_model() -> None:
    global _model, _model_version, _thresholds
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        _model = mlflow.catboost.load_model(model_uri)
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        _model_version = latest[0].version if latest else "unknown"
        MODEL_VERSION_GAUGE.labels(stage=MODEL_STAGE).set(int(_model_version) if _model_version.isdigit() else 0)
        print(f"loaded model {MODEL_NAME}@{MODEL_STAGE} version={_model_version}")
    except Exception as e:
        # На старте не валимся, чтобы /health мог отрапортовать not_ready
        print(f"WARN: model load failed: {e}")

    if THRESHOLDS_PATH.exists():
        _thresholds = json.loads(THRESHOLDS_PATH.read_text())
    else:
        _thresholds = {"auto_merge": 0.998, "review": 0.95}


# --------------------------------------------------------------------------- #
# Схемы
# --------------------------------------------------------------------------- #
class Pair(BaseModel):
    profile_id_1: str
    profile_id_2: str
    features: dict[str, Any]   # match_*, diff_*, jaccard_*, overlap_*

class ScoreRequest(BaseModel):
    pairs: list[Pair]

class ScoreResponse(BaseModel):
    model_version: str
    results: list[dict]


# --------------------------------------------------------------------------- #
# Эндпоинты
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    if _model is None:
        raise HTTPException(503, detail="model not loaded")
    return {"status": "ok", "model_version": _model_version, "stage": MODEL_STAGE}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if _model is None:
        REQUESTS.labels(status="error", stage=MODEL_STAGE).inc()
        raise HTTPException(503, detail="model not ready")

    t0 = time.perf_counter()
    df = pd.DataFrame([p.features for p in req.pairs])
    cat_features = [c for c in df.columns if c.startswith("match_")]
    for c in cat_features:
        df[c] = df[c].astype(int).astype(str)

    try:
        prob = _model.predict_proba(df)[:, 1]
    except Exception as e:
        REQUESTS.labels(status="error", stage=MODEL_STAGE).inc()
        raise HTTPException(500, detail=f"inference failed: {e}") from e

    auto_thr = _thresholds.get("auto_merge", 0.998)
    review_thr = _thresholds.get("review", 0.95)

    results = []
    for pair, p in zip(req.pairs, prob):
        action = "AUTO_MERGE" if p >= auto_thr else ("REVIEW" if p >= review_thr else "KEEP_SEPARATE")
        ACTIONS.labels(action=action).inc()
        results.append({
            "profile_id_1": pair.profile_id_1,
            "profile_id_2": pair.profile_id_2,
            "prob_match": float(p),
            "action": action,
        })

    PAIRS_SCORED.inc(len(results))
    PROB_MEAN.set(float(np.mean(prob)))
    REQUESTS.labels(status="ok", stage=MODEL_STAGE).inc()
    LATENCY.observe(time.perf_counter() - t0)
    return ScoreResponse(model_version=_model_version, results=results)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
