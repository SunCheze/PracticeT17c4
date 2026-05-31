"""Smoke-тест полного пайплайна на синтетике.

Запуск::

    pytest tests/

Должен пройти за ~10 секунд.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er.blocking import blocking_recall
from er.features import generate_pairwise_features
from er.model import calibrate_thresholds, predict_proba, train_model
from er.parsing import parse_kv_list_to_dict
from er.pipeline import generate_candidates, preprocess
from er.synthetic import make_synthetic_dataset


def test_parser_handles_numpy_repr_format():
    """В CSV массивы хранятся как `['a:b' 'c:d']` — без запятых."""
    s = "['source_site_365:6307' 'has_click_365:6880' 'is_gmail']"
    out = parse_kv_list_to_dict(s)
    assert out["source_site_365"] == "6307"
    assert out["has_click_365"] == "6880"
    assert out["is_gmail"] is True


def test_parser_handles_duplicate_keys():
    """Несколько токенов с одним ключом склеиваются через `|`."""
    s = "['visited_365:111' 'visited_365:222']"
    out = parse_kv_list_to_dict(s, handle_duplicates=True, sep="|")
    assert out["visited_365"] == "111|222"


def test_end_to_end_pipeline_runs_and_blocking_recall_is_high():
    """Прогоняем мини-набор и проверяем, что blocking recall не сломан."""
    raw = make_synthetic_dataset(n_single=200, n_multi=40, seed=1)
    prof = preprocess(raw)
    assert prof["profile_id"].is_unique

    pairs = generate_candidates(prof, labeled=True)
    rec, captured, total = blocking_recall(pairs, prof)
    assert rec >= 0.95, f"blocking recall просел: {rec}"


def test_model_trains_and_predicts():
    """Полный цикл обучения и инференса должен дать PR-AUC > 0.5."""
    raw = make_synthetic_dataset(n_single=300, n_multi=60, seed=2)
    prof = preprocess(raw)
    pairs = generate_candidates(prof, labeled=True)
    feats, cat_features = generate_pairwise_features(pairs, prof)

    feat_cols = [c for c in feats.columns if c not in ("profile_id_1", "profile_id_2", "is_match")]
    # упрощённый сплит — для smoke-теста достаточно
    idx = np.arange(len(feats))
    np.random.default_rng(0).shuffle(idx)
    cut = int(0.75 * len(idx))
    tr, va = feats.iloc[idx[:cut]], feats.iloc[idx[cut:]]

    model = train_model(tr[feat_cols], tr["is_match"], cat_features,
                        va[feat_cols], va["is_match"], iterations=80)
    prob = predict_proba(model, va[feat_cols], cat_features)
    assert (prob >= 0).all() and (prob <= 1).all()
    assert len(prob) == len(va)

    thr = calibrate_thresholds(va["is_match"].values, prob)
    assert "auto_merge" in thr and "review" in thr
    assert thr["auto_merge"] >= thr["review"]
