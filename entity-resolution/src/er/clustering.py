"""Кластеризация пар в связные компоненты и рекомендация действий.

После предсказания вероятностей P(match) для каждой пары:
- пары с p ≥ THRESHOLD_AUTO_MERGE  → AUTO-MERGE (объединить)
- пары с p ≥ THRESHOLD_REVIEW       → REVIEW (отправить оператору)
- остальные                          → KEEP_SEPARATE
Связные компоненты графа пар собираются в кластеры дубликатов.
"""
from __future__ import annotations
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from .config import THRESHOLD_AUTO_MERGE, THRESHOLD_REVIEW

Action = Literal["AUTO_MERGE", "REVIEW", "KEEP_SEPARATE"]


def recommend_action(prob: float, auto_thr: float = THRESHOLD_AUTO_MERGE,
                     review_thr: float = THRESHOLD_REVIEW) -> Action:
    if prob >= auto_thr:
        return "AUTO_MERGE"
    if prob >= review_thr:
        return "REVIEW"
    return "KEEP_SEPARATE"


def cluster_pairs(
    pairs: pd.DataFrame,
    prob: np.ndarray,
    all_profile_ids: list[str] | None = None,
    auto_thr: float = THRESHOLD_AUTO_MERGE,
    review_thr: float = THRESHOLD_REVIEW,
) -> pd.DataFrame:
    """Возвращает DataFrame со столбцом cluster_id для каждого профиля.

    Использует только AUTO-рёбра для связных компонент (консервативная стратегия):
    REVIEW-рёбра НЕ объединяют автоматически — они уходят в очередь оператору.

    Параметры:
        pairs: DataFrame с колонками profile_id_1, profile_id_2
        prob:  массив вероятностей такой же длины
        all_profile_ids: полный набор профилей (для изолированных кластеров)
    """
    df = pairs[["profile_id_1", "profile_id_2"]].copy()
    df["prob"] = prob
    df["action"] = df["prob"].apply(
        lambda p: recommend_action(p, auto_thr, review_thr)
    )

    G = nx.Graph()
    if all_profile_ids is not None:
        G.add_nodes_from(all_profile_ids)
    else:
        G.add_nodes_from(pd.concat([df["profile_id_1"], df["profile_id_2"]]).unique())

    auto_edges = df.loc[df["action"] == "AUTO_MERGE", ["profile_id_1", "profile_id_2"]].values
    G.add_edges_from(auto_edges)

    cluster_map: dict[str, int] = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            cluster_map[n] = cid

    profiles = pd.DataFrame({
        "profile_id": list(cluster_map.keys()),
        "cluster_id": list(cluster_map.values()),
    })
    return profiles


def summarize_clusters(profiles: pd.DataFrame, pairs_with_action: pd.DataFrame) -> pd.DataFrame:
    """Сводка по кластерам: размер, истинные/предсказанные размеры, рекомендации."""
    sizes = profiles.groupby("cluster_id").size().rename("size")
    return sizes.reset_index().sort_values("size", ascending=False)
