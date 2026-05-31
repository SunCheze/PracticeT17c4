"""Попарные признаки для модели матчинга.

Категории:
- match_*  — трёхзначные категориальные совпадения PII/категорий (-1=NA, 0=разные, 1=совпали).
- diff_*   — числовые расстояния (даты, числовые поля).
- jaccard_* / overlap_* — пересечения site-id (поведенческий сигнал).
- overlap_any_site — агрегат поведенческого пересечения, главный драйвер модели.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import SITE_SET_COLS


def _state3(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Трёхзначное состояние совпадения с явной обработкой NaN."""
    a = a.astype(object).where(a.notna(), None).values
    b = b.astype(object).where(b.notna(), None).values
    r = np.full(len(a), -1, dtype=int)
    for i in range(len(a)):
        if a[i] is None or b[i] is None:
            continue
        r[i] = 1 if a[i] == b[i] else 0
    return r


def _jacc(s1, s2) -> float:
    if not s1 or not s2:
        return -1.0
    inter = len(s1 & s2)
    return inter / len(s1 | s2) if inter else 0.0


def _overlap(s1, s2) -> int:
    if not s1 or not s2:
        return -1
    return len(s1 & s2)


def generate_pairwise_features(
    pairs_df: pd.DataFrame, prof: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Возвращает (DataFrame признаков, список категориальных колонок)."""
    cat_cols = [
        c for c in [
            "first_name", "sex", "np_device", "np_browser", "np_osfamily",
            "np_subdivision_1_iso_code", "rt_country", "rt_is_million",
            "fs_is_man", "fs_is_woman", "fs_is_gmail", "fs_postman_response_90",
        ] if c in prof.columns
    ]
    id_cols = [c for c in ["np_geoname_id", "rt_geoid"] if c in prof.columns]
    num_cols = [c for c in ["rt_tz_offset", "rt_local_hour", "rt_day", "rt_population"] if c in prof.columns]
    set_cols = [c for c in SITE_SET_COLS if c in prof.columns]
    extra = ["email_domain"] if "email_domain" in prof.columns else []
    needed = cat_cols + num_cols + id_cols + set_cols + extra + ["created_at"]

    b = prof[["profile_id"] + needed].drop_duplicates("profile_id")
    d1 = b.add_suffix("_1").rename(columns={"profile_id_1": "profile_id_1"})
    d2 = b.add_suffix("_2").rename(columns={"profile_id_2": "profile_id_2"})

    m = pairs_df.merge(d1, on="profile_id_1", how="left").merge(d2, on="profile_id_2", how="left")

    keep = ["profile_id_1", "profile_id_2"] + (["is_match"] if "is_match" in pairs_df.columns else [])
    res = pairs_df[keep].copy().reset_index(drop=True)

    for c in cat_cols + id_cols:
        res[f"match_{c}"] = _state3(m[f"{c}_1"], m[f"{c}_2"])
    if extra:
        res["match_email_domain"] = _state3(m["email_domain_1"], m["email_domain_2"])

    for c in num_cols:
        c1 = pd.to_numeric(m[f"{c}_1"], errors="coerce")
        c2 = pd.to_numeric(m[f"{c}_2"], errors="coerce")
        res[f"diff_{c}"] = (c1 - c2).abs()

    dt1 = pd.to_datetime(m["created_at_1"], errors="coerce")
    dt2 = pd.to_datetime(m["created_at_2"], errors="coerce")
    res["diff_created_days"] = (dt1 - dt2).dt.days.abs()

    for c in set_cols:
        c1, c2 = m[f"{c}_1"].values, m[f"{c}_2"].values
        res[f"jaccard_{c}"] = [_jacc(a, b) for a, b in zip(c1, c2)]
        res[f"overlap_{c}"] = [_overlap(a, b) for a, b in zip(c1, c2)]

    res["overlap_any_site"] = res[[f"overlap_{c}" for c in set_cols]].clip(lower=0).sum(axis=1)
    cat_features = [c for c in res.columns if c.startswith("match_")]
    return res, cat_features
