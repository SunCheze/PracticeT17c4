"""Синтетический генератор данных под реальную схему.

Используется для тестов и демо без приватных данных. Не запускайте на
проде — это только для smoke-тестов пайплайна.
"""
from __future__ import annotations
import json
import random

import numpy as np
import pandas as pd

DOMAINS = ["gmail.com", "mail.ru", "yandex.ru", "bk.ru", "msil.ru", "inbox.ru"]
CITIES = [
    ("Moscow", 524901, 3, 10381222, True),
    ("Chelyabinsk", 1508291, 5, 1202371, True),
    ("Kazan", 551487, 3, 1243500, True),
    ("Yekaterinburg", 1486209, 5, 1495066, True),
    ("Nizhnekamsk", 521118, 3, 234297, False),
    ("Tashkent", 1512569, 5, 1978028, True),
]
DEVICES = ["smartphone", "pc", "tablet"]
BROWSERS = ["chrome", "yandex", "safari", "firefox"]
OS = ["android", "ios", "windows"]
NAMES_F = ["Ксения", "Любовь", "Анна", "Мария", "Ольга"]
NAMES_M = ["Артём", "Иван", "Пётр", "Сергей", "Дмитрий"]


def _rand_email(domain: str) -> str:
    loc = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(6, 14)))
    if random.random() < 0.4:
        loc += str(random.randint(0, 9999))
    return f"{loc}@{domain}"


def _np_array_repr(items: list) -> str:
    """Воспроизводит формат numpy-str-массива в CSV: single quotes, токены через пробел."""
    if not items:
        return "[]"
    return "[" + " ".join(f"'{x}'" for x in items) + "]"


def _make_fs(site_pool, sex, include_all=False):
    items = []
    chosen = list(site_pool) if include_all else random.sample(site_pool, random.randint(1, min(8, len(site_pool))))
    for sid in chosen:
        pref = random.choice(["visited_365", "visited_30", "has_account", "has_click_365",
                              "source_site_365", "has_accept_365"])
        items.append(f"{pref}:{sid}")
    if random.random() < 0.5:
        items.append("is_gmail")
    if sex == "male":
        items.append("is_man")
    elif sex == "female":
        items.append("is_woman")
    if random.random() < 0.3:
        items.append(f"postman_response_90:{random.choice(['ok', 'err'])}")
    return _np_array_repr(items)


def _make_np(city):
    name, geoid, tz, pop, ismil = city
    iso = random.choice(["MOW", "CHE", "TA", "SVE", "TK", "GB"])
    items = [
        f"browser:{random.choice(BROWSERS)}",
        f"subdivision_1_iso_code:{iso}",
        f"device:{random.choice(DEVICES)}",
        f"geoname_id:{geoid}",
        f"osfamily:{random.choice(OS)}",
    ]
    return _np_array_repr(items)


def _make_rt(city):
    name, geoid, tz, pop, ismil = city
    d = {"country": "RU", "is_million": bool(ismil), "tz_offset": int(tz),
         "geoname": name, "geoid": int(geoid),
         "local_hour": random.randint(0, 23), "day": random.randint(0, 6),
         "population": int(pop)}
    if random.random() < 0.4:
        d["visit_count"] = str(random.randint(1, 30))
    return json.dumps(d)


def make_synthetic_dataset(n_single: int = 2500, n_multi: int = 350, seed: int = 42) -> pd.DataFrame:
    """Сгенерировать датасет уровня событий. Один профиль может встречаться
    в нескольких строках. Мульти-сущности делят общий пул site-id и
    ~в 45% случаев — два разных email-домена (кросс-доменные дубли)."""
    random.seed(seed)
    rows = []

    def emit(entity_id, site_pool, base_city, base_name, base_sex, domain, include_all=False):
        profile_id = f"p_{random.getrandbits(60):015x}"
        sex = base_sex if random.random() < 0.85 else "unknown"
        first = base_name if (base_name and random.random() < 0.6) else None
        email = _rand_email(domain) if random.random() < 0.98 else None
        phone = ("79" + str(random.choice([520, 521, 916, 903])) + f"{random.randint(0, 9999999):07d}"
                 ) if random.random() < 0.08 else None
        bday = None
        if random.random() < 0.05:
            bday = pd.Timestamp(f"19{random.randint(60,99)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
        n_events = random.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
        for _ in range(n_events):
            city = base_city if random.random() < 0.8 else random.choice(CITIES)
            rows.append({
                "created_at": pd.Timestamp("2025-11-01") + pd.Timedelta(days=random.randint(0, 200),
                                                                        seconds=random.randint(0, 86399)),
                "first_name": first, "last_name": None, "email": email, "phone": phone,
                "birthday": bday, "sex": sex,
                "non_processing_features": _make_np(city),
                "realtime_features": _make_rt(city),
                "fs_features": _make_fs(site_pool, sex, include_all=include_all),
                "profile_id": profile_id, "entity_id": entity_id,
            })

    for i in range(n_single):
        eid = f"e_s_{i:06d}"
        pool = random.sample(range(1000, 7000), random.randint(1, 6))
        city = random.choice(CITIES)
        sex = random.choice(["male", "female"])
        name = random.choice(NAMES_M if sex == "male" else NAMES_F)
        emit(eid, pool, city, name, sex, random.choice(DOMAINS))

    for i in range(n_multi):
        eid = f"e_m_{i:06d}"
        shared_pool = random.sample(range(1000, 7000), random.randint(3, 8))
        city = random.choice(CITIES)
        sex = random.choice(["male", "female"])
        name = random.choice(NAMES_M if sex == "male" else NAMES_F)
        nprof = random.choices([2, 3, 4], weights=[0.7, 0.2, 0.1])[0]
        doms = random.sample(DOMAINS, 2) if random.random() < 0.45 else [random.choice(DOMAINS)]
        for j in range(nprof):
            pool = list(set(shared_pool) | set(random.sample(range(1000, 7000), random.randint(0, 2))))
            emit(eid, pool, city, name, sex, doms[j % len(doms)], include_all=True)

    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
