"""Парсинг вложенных полей.

Поля ``non_processing_features`` / ``realtime_features`` / ``fs_features``
приходят как массивы или JSON-строки, в CSV — как repr numpy-массива
(токены в одинарных кавычках через пробел, БЕЗ запятых).
``ast.literal_eval`` на таком формате молча конкатенирует соседние
строковые литералы — поэтому здесь используется regex.
"""
from __future__ import annotations
import ast
import json
import re
from typing import Any

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _coerce_to_list(s: Any) -> list[str]:
    """Получить список строковых токенов из list / ndarray / numpy-repr / NaN."""
    if isinstance(s, (list, tuple, np.ndarray)):
        return [x for x in s if isinstance(x, str)]
    if isinstance(s, str):
        st = s.strip()
        if st in ("", "[]", "{}"):
            return []
        if st.startswith("["):
            toks = [a or b for a, b in _TOKEN_RE.findall(st)]
            if toks:
                return toks
            try:
                v = ast.literal_eval(st)
                return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []
            except (ValueError, SyntaxError):
                return []
    return []


def parse_kv_list_to_dict(s: Any, handle_duplicates: bool = True, sep: str = "|") -> dict:
    """Превратить список токенов 'key:value' / 'flag' в словарь.

    handle_duplicates=True склеивает повторы через ``sep`` (нужно для fs_features,
    где у одного профиля может быть несколько `visited_365:<id>`).
    """
    items = _coerce_to_list(s)
    result: dict = {}
    for item in items:
        if not isinstance(item, str):
            continue
        if ":" in item:
            k, v = item.split(":", 1)
        else:
            k, v = item, True  # флаг без значения
        if handle_duplicates and k in result:
            result[k] = f"{result[k]}{sep}{v}"
        elif not handle_duplicates and k in result:
            continue
        else:
            result[k] = v
    return result


def parse_json_to_dict(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        st = s.strip()
        if st in ("", "{}"):
            return {}
        try:
            r = json.loads(st)
            return r if isinstance(r, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}
    return {}


def flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Развернуть np_/rt_/fs_ поля в плоские колонки."""
    np_dicts = df["non_processing_features"].apply(
        lambda x: parse_kv_list_to_dict(x, handle_duplicates=False)
    )
    rt_dicts = df["realtime_features"].apply(parse_json_to_dict)
    fs_dicts = df["fs_features"].apply(
        lambda x: parse_kv_list_to_dict(x, handle_duplicates=True, sep="|")
    )
    np_df = pd.json_normalize(np_dicts).add_prefix("np_")
    rt_df = pd.json_normalize(rt_dicts).add_prefix("rt_")
    fs_df = pd.json_normalize(fs_dicts).add_prefix("fs_")
    base = df.drop(
        columns=["non_processing_features", "realtime_features", "fs_features"]
    ).reset_index(drop=True)
    return pd.concat(
        [base, np_df.reset_index(drop=True), rt_df.reset_index(drop=True), fs_df.reset_index(drop=True)],
        axis=1,
    )
