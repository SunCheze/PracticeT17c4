# Entity Resolution: поиск дубликатов клиентских профилей

Решение задачи Entity Resolution для маркетплейса: один реальный человек
заводит несколько профилей (с разными email, опечатками в ФИО, отсутствующими
полями), и нужно автоматически их объединить.

**Архитектура:** парсинг → схлопывание событий → мульти-ключевой блокинг
→ попарные признаки → CatBoost → кластеризация в связные компоненты
→ два порога решения (AUTO-MERGE / REVIEW).

---

## Быстрый старт (3 команды)

```bash
git clone https://github.com/<user>/entity-resolution.git
cd entity-resolution
pip install -e .                          # установить пакет в editable-режиме

python -m scripts.train --synthetic       # обучить на синтетике (~30 сек)
streamlit run app/streamlit_app.py        # запустить демо
```

Это поднимает работающий пайплайн от и до. Дальше можно подсунуть свои данные.

---

## Структура репозитория

```
entity-resolution/
├── README.md
├── pyproject.toml             # зависимости + точки входа CLI
├── requirements.txt           # альтернатива pyproject.toml
├── .gitignore
├── src/er/                    # ПАКЕТ — вся бизнес-логика
│   ├── config.py              # пути, пороги, имена колонок
│   ├── parsing.py             # парсинг np-массивов в CSV
│   ├── preprocessing.py       # схлопывание событий в профили
│   ├── blocking.py            # мульти-ключевой блокинг
│   ├── features.py            # попарные признаки
│   ├── model.py               # обучение/инференс CatBoost
│   ├── clustering.py          # связные компоненты + рекомендации
│   ├── pipeline.py            # сквозная оркестрация
│   └── synthetic.py           # генератор для тестов и демо
├── scripts/                   # CLI-обвязка
│   ├── train.py               # обучение модели
│   └── predict.py             # инференс на батче
├── app/
│   └── streamlit_app.py       # UI для демонстрации
├── tests/
│   └── test_smoke.py          # smoke-тест end-to-end
├── notebooks/                 # исследовательские ноутбуки
├── data/
│   ├── raw/                   # сырые данные (gitignored)
│   └── processed/             # выгрузки скоров (gitignored)
└── models/                    # артефакты (gitignored)
```

**Главный принцип:** в `src/er/` лежат функции, в `scripts/` и `app/` — точки входа,
которые эти функции вызывают. Логика не дублируется.

---

## Установка

Требуется Python 3.10+.

```bash
# Вариант 1: editable-установка как пакета (рекомендуется)
pip install -e .

# Вариант 2: только зависимости, без установки пакета
pip install -r requirements.txt
```

После `pip install -e .` появятся CLI-команды `er-train` и `er-predict`.

---

## Формат входных данных

Один файл (parquet или csv) со столбцами:

| Колонка | Тип | Описание |
|---|---|---|
| `created_at` | timestamp | время события |
| `first_name`, `last_name` | str | ФИО (может быть NaN) |
| `email`, `phone`, `birthday`, `sex` | str | PII (может быть NaN) |
| `non_processing_features` | str | numpy-repr массива `[...]` |
| `realtime_features` | str | JSON-строка |
| `fs_features` | str | numpy-repr массива `[...]` |
| `profile_id` | str | ID профиля |
| `entity_id` | str | ID сущности (нужен только для обучения) |

Один профиль может встречаться в нескольких строках-событиях — пайплайн их схлопывает.

---

## Использование

### Обучение

```bash
# на реальных данных
python -m scripts.train --data data/raw/profiles.parquet

# на синтетике (для smoke-теста)
python -m scripts.train --synthetic

# с параметрами
python -m scripts.train --data data/raw/profiles.parquet \
    --val-fraction 0.2 --neg-ratio 30 --target-precision-auto 0.99
```

На выходе: `models/catboost_er.cbm` + `models/thresholds.json`.

### Инференс

```bash
python -m scripts.predict --input data/raw/new_batch.parquet \
    --output-pairs data/processed/pairs.csv \
    --output-clusters data/processed/clusters.csv
```

Получаются два файла:
- `pairs.csv` — пары с вероятностью совпадения и рекомендуемым действием
- `clusters.csv` — `profile_id → cluster_id`

### Демо-интерфейс (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Откроется браузер на `http://localhost:8501`. Можно загрузить файл или сгенерировать
синтетику, увидеть найденные дубли с вероятностями и рекомендациями, посмотреть
кластеры, скачать результат.

### Программный API

```python
from er.pipeline import load_raw, run_inference

raw = load_raw("data/raw/profiles.parquet")
result = run_inference(raw, model_path="models/catboost_er.cbm")
print(result["summary"])
print(result["pairs_scored"].head())
```

---

## Тесты и проверка воспроизводимости

```bash
pytest tests/                     # smoke-тест (~10 секунд)
```

Полный sanity-check на чистом окружении (то, что должен пройти любой клон):

```bash
git clone https://github.com/<user>/entity-resolution.git tmp_check
cd tmp_check
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"
pytest tests/
python -m scripts.train --synthetic
python -m scripts.predict --input <ваш файл> --output-pairs /tmp/p.csv --output-clusters /tmp/c.csv
```

Если все четыре шага проходят — репо воспроизводим.

---

## Архитектура решения

```
Сырые события  →  flatten()           →  плоские колонки np_/rt_/fs_
                  collapse_to_profiles()→  1 строка на profile_id

Профили        →  build_blocks()      →  ключи: dom::, phone::, name::, site::
                  candidate_pairs()    →  множество неупорядоченных пар

Пары           →  generate_pairwise_features()
                                       →  match_* (PII), diff_*, jaccard_*, overlap_*

Признаки       →  CatBoost.predict_proba() →  P(match) ∈ [0, 1]

Вероятности    →  recommend_action()    →  AUTO_MERGE / REVIEW / KEEP_SEPARATE
                  cluster_pairs()       →  связные компоненты для AUTO-пар
```

Подробное обсуждение и метрики качества — в техническом отчёте проекта.

---

## Метрики (на синтетике; на реальных данных могут отличаться)

| Метрика | AUTO-MERGE | REVIEW |
|---|---|---|
| Порог P(match) | 0.998 | 0.966 |
| Pair Precision / Recall / F1 | 1.00 / 0.10 / 0.17 | 0.85 / 0.67 / 0.75 |
| B-Cubed F1 (кластеры) | 0.932 | 0.966 |
| PR-AUC | 0.733 | 0.733 |
| Blocking recall | 100% | 100% |

---

## Развитие

- **MinHash/LSH для site-id** на больших объёмах (`datasketch` уже совместим с пайплайном).
- **Запуск на Spark** (`pyspark` для блокинга и признаков).
- **ONNX-экспорт** CatBoost для онлайн-инференса в Triton.
- **Continuous learning** — метки из REVIEW-очереди в обучающую выборку.

См. технический отчёт в /docs репозитория.
