"""Схлопывание событий в один профиль.

Один профиль может встречаться в нескольких строках-событиях; склеиваем
в одну строку на ``profile_id``: для скалярных полей — последнее событие,
для site-id — объединение всех значений.
"""
from __future__ import annotations
from collections import defaultdict

import pandas as pd

from .config import SITE_SET_COLS


def collapse_to_profiles(flat: pd.DataFrame) -> pd.DataFrame:
    flat = flat.copy()
    flat["created_at"] = pd.to_datetime(flat["created_at"], errors="coerce")
    flat = flat.sort_values("created_at")

    present_site_cols = [c for c in SITE_SET_COLS if c in flat.columns]
    union_map = {c: defaultdict(set) for c in present_site_cols}
    for c in present_site_cols:
        sub = flat[["profile_id", c]].dropna(subset=[c])
        for pid, val in zip(sub["profile_id"].values, sub[c].values):
            if isinstance(val, str) and val:
                union_map[c][pid].update(val.split("|"))

    prof = flat.groupby("profile_id", as_index=False).last()  # последнее событие на профиль
    for c in present_site_cols:
        prof[c] = prof["profile_id"].map(
            lambda p: frozenset(union_map[c].get(p, set())) or None
        )

    prof["email_domain"] = prof["email"].astype("string").str.split("@").str[-1]
    return prof
