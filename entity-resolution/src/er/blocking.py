"""Мульти-ключевой блокинг и генерация пар-кандидатов.

Ключи: домен email, телефон, имя × geoid, общий site-id из fs_features.
Поведенческий ключ ловит кросс-доменные дубли, которые теряет блокинг только по домену.
"""
from __future__ import annotations
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import SITE_SET_COLS, BLOCK_MAX_SIZE, SITE_STOP_MIN_DF


def build_blocks(
    prof: pd.DataFrame,
    max_block: int = BLOCK_MAX_SIZE,
    site_stop_min_df: int = SITE_STOP_MIN_DF,
    rng: np.random.Generator | None = None,
) -> dict[str, list[str]]:
    """Возвращает dict: block_key -> список profile_id.

    Слишком частые site-id (df > site_stop_min_df) считаются стоп-словами и отбрасываются.
    Большие блоки down-sample'ятся до ``max_block``, чтобы число пар не взорвалось.
    """
    rng = rng or np.random.default_rng(0)
    blocks: dict[str, list[str]] = defaultdict(list)
    cols = prof.columns
    site_cols = [c for c in SITE_SET_COLS if c in cols]

    for _, r in prof.iterrows():
        pid = r["profile_id"]
        if pd.notna(r.get("email_domain")):
            blocks[f"dom::{r['email_domain']}"].append(pid)
        if pd.notna(r.get("phone")):
            blocks[f"phone::{r['phone']}"].append(pid)
        fn = r.get("first_name")
        gid = r.get("rt_geoid") if "rt_geoid" in cols else None
        if pd.notna(fn) and pd.notna(gid):
            blocks[f"name::{fn}::{gid}"].append(pid)
        all_sites: set[str] = set()
        for c in site_cols:
            s = r.get(c)
            if s:
                all_sites |= set(s)
        for tok in all_sites:
            blocks[f"site::{tok}"].append(pid)

    clean: dict[str, list[str]] = {}
    for k, pids in blocks.items():
        u = list(dict.fromkeys(pids))
        if len(u) < 2:
            continue
        if k.startswith("site::") and len(u) > site_stop_min_df:
            continue
        if len(u) > max_block:
            u = list(rng.choice(u, size=max_block, replace=False))
        clean[k] = u
    return clean


def candidate_pairs(blocks: dict[str, list[str]]) -> pd.DataFrame:
    """Уникальные неупорядоченные пары по всем блокам."""
    seen: set[tuple[str, str]] = set()
    for pids in blocks.values():
        for a, b in itertools.combinations(sorted(pids), 2):
            seen.add((a, b))
    return pd.DataFrame(list(seen), columns=["profile_id_1", "profile_id_2"])


def label_pairs(pairs: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку is_match по совпадению entity_id (нужно только для оценки качества)."""
    e = prof.set_index("profile_id")["entity_id"]
    pairs = pairs.copy()
    pairs["is_match"] = (
        pairs["profile_id_1"].map(e).values == pairs["profile_id_2"].map(e).values
    ).astype(int)
    return pairs


def blocking_recall(pairs: pd.DataFrame, prof: pd.DataFrame) -> tuple[float, int, int]:
    """Доля истинных пар-дублей, пойманных блокингом."""
    ent_groups = prof.groupby("entity_id")["profile_id"].apply(list)
    truth: set[tuple[str, str]] = set()
    for pids in ent_groups:
        if len(pids) >= 2:
            for a, b in itertools.combinations(sorted(pids), 2):
                truth.add((a, b))
    if not truth:
        return 1.0, 0, 0
    got = set(
        map(
            tuple,
            pairs.loc[pairs["is_match"] == 1, ["profile_id_1", "profile_id_2"]].values,
        )
    )
    captured = len(truth & got)
    return captured / len(truth), captured, len(truth)
