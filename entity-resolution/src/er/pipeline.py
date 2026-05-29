"""Сквозной пайплайн от сырых данных до кластеров дубликатов.

Используется и из CLI-скриптов (scripts/train.py, scripts/predict.py),
и из Streamlit-UI. Чтобы не было дублирования логики.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import build_blocks, candidate_pairs, label_pairs, blocking_recall
from .clustering import cluster_pairs, recommend_action
from .features import generate_pairwise_features
from .model import load_model, predict_proba
from .parsing import flatten
from .preprocessing import collapse_to_profiles


def load_raw(path: Path | str) -> pd.DataFrame:
    """Загрузка parquet/csv с сырыми данными.

    Ожидаемые колонки: created_at, first_name, last_name, email, phone, birthday, sex,
    non_processing_features, realtime_features, fs_features, profile_id, entity_id (опц.).
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    raise ValueError(f"Неизвестный формат: {path.suffix}")


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Сырые события -> один профиль на строку."""
    return collapse_to_profiles(flatten(raw))


def generate_candidates(prof: pd.DataFrame, labeled: bool = False) -> pd.DataFrame:
    """Профили -> пары-кандидаты (опционально с разметкой по entity_id)."""
    blocks = build_blocks(prof, rng=np.random.default_rng(0))
    pairs = candidate_pairs(blocks)
    if labeled and "entity_id" in prof.columns:
        pairs = label_pairs(pairs, prof)
    return pairs


def score_pairs(pairs: pd.DataFrame, prof: pd.DataFrame, model_path: Path | str) -> pd.DataFrame:
    """Полный шаг инференса: признаки -> вероятности -> рекомендации."""
    feats, cat_features = generate_pairwise_features(pairs, prof)
    feat_cols = [c for c in feats.columns
                 if c not in ("profile_id_1", "profile_id_2", "is_match")]
    model = load_model(model_path)
    prob = predict_proba(model, feats[feat_cols], cat_features)
    out = pairs[["profile_id_1", "profile_id_2"]].copy()
    out["prob_match"] = prob
    out["action"] = out["prob_match"].apply(recommend_action)
    return out


def run_inference(
    raw: pd.DataFrame,
    model_path: Path | str,
) -> dict:
    """Полный инференс на батче профилей. Возвращает:
        {prof, pairs_scored, clusters, summary}
    где prof — схлопнутые профили, pairs_scored — пары с вероятностями и действиями,
    clusters — DataFrame profile_id -> cluster_id, summary — статистика.
    """
    prof = preprocess(raw)
    pairs = generate_candidates(prof, labeled=False)

    if len(pairs) == 0:
        clusters = pd.DataFrame({"profile_id": prof["profile_id"], "cluster_id": range(len(prof))})
        return dict(
            prof=prof,
            pairs_scored=pairs,
            clusters=clusters,
            summary={"n_profiles": len(prof), "n_pairs": 0, "n_auto": 0, "n_review": 0},
        )

    pairs_scored = score_pairs(pairs, prof, model_path)
    clusters = cluster_pairs(
        pairs_scored, pairs_scored["prob_match"].values,
        all_profile_ids=prof["profile_id"].tolist(),
    )
    summary = {
        "n_profiles": int(len(prof)),
        "n_pairs": int(len(pairs_scored)),
        "n_auto": int((pairs_scored["action"] == "AUTO_MERGE").sum()),
        "n_review": int((pairs_scored["action"] == "REVIEW").sum()),
        "n_clusters_with_dups": int(
            clusters.groupby("cluster_id").size().gt(1).sum()
        ),
    }
    return dict(prof=prof, pairs_scored=pairs_scored, clusters=clusters, summary=summary)
