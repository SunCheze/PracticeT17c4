"""Streamlit-демо для Entity Resolution.

Запуск::

    streamlit run app/streamlit_app.py

Что умеет:
  1. Загрузка батча профилей (csv/parquet) или генерация синтетики.
  2. Запуск полного пайплайна (предобработка → блокинг → модель → кластеризация).
  3. Просмотр найденных дублей с вероятностями и рекомендациями.
  4. Скачивание результата.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Добавляем src/ в путь (когда запускаем из корня репо)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er.config import (DEFAULT_MODEL_PATH, DEFAULT_THRESHOLDS_PATH,
                       THRESHOLD_AUTO_MERGE, THRESHOLD_REVIEW)
from er.model import load_thresholds
from er.pipeline import load_raw, run_inference
from er.synthetic import make_synthetic_dataset

# ---- Setup ----
st.set_page_config(
    page_title="Entity Resolution — поиск дубликатов",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔗 Entity Resolution — поиск дубликатов профилей")
st.caption("Загрузите батч профилей — система найдёт дубли и предложит, что с ними делать.")

# ---- Sidebar: настройки ----
with st.sidebar:
    st.header("⚙️ Параметры")

    # Пытаемся подгрузить откалиброванные пороги из models/
    try:
        thr = load_thresholds(DEFAULT_THRESHOLDS_PATH)
        default_auto = float(thr.get("auto_merge", THRESHOLD_AUTO_MERGE))
        default_review = float(thr.get("review", THRESHOLD_REVIEW))
        st.caption(f"Загружены откалиброванные пороги (PR-AUC = {thr.get('pr_auc', 0):.3f})")
    except FileNotFoundError:
        default_auto, default_review = THRESHOLD_AUTO_MERGE, THRESHOLD_REVIEW
        st.caption("Используются дефолтные пороги (модель не калибрована)")

    auto_thr = st.slider("Порог AUTO-MERGE", 0.5, 1.0, default_auto, 0.001, format="%.3f")
    review_thr = st.slider("Порог REVIEW", 0.5, 1.0, default_review, 0.001, format="%.3f")
    if review_thr >= auto_thr:
        st.warning("Порог REVIEW должен быть ниже AUTO-MERGE")

    st.divider()
    model_path = st.text_input("Путь к модели", str(DEFAULT_MODEL_PATH))
    if not Path(model_path).exists():
        st.error(f"Модель не найдена. Сначала обучите её:\n`python -m scripts.train --synthetic`")
        st.stop()

# ---- Источник данных ----
st.subheader("1. Источник данных")
source = st.radio("Откуда брать профили?",
                   ["Загрузить файл", "Сгенерировать синтетику (для демо)"],
                   horizontal=True)

raw_df: pd.DataFrame | None = None
if source == "Загрузить файл":
    uploaded = st.file_uploader("CSV или Parquet с сырыми профилями", type=["csv", "parquet"])
    if uploaded:
        try:
            if uploaded.name.endswith(".parquet"):
                raw_df = pd.read_parquet(uploaded)
            else:
                raw_df = pd.read_csv(uploaded)
            st.success(f"Загружено: {len(raw_df):,} строк")
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
else:
    n_single = st.slider("Одиночных сущностей", 100, 3000, 500, 100)
    n_multi = st.slider("Мульти-сущностей (с дублями)", 10, 500, 80, 10)
    if st.button("🎲 Сгенерировать", type="primary"):
        with st.spinner("Генерируем..."):
            raw_df = make_synthetic_dataset(n_single=n_single, n_multi=n_multi)
        st.success(f"Сгенерировано: {len(raw_df):,} строк ({n_single + n_multi} сущностей)")

if raw_df is None:
    st.stop()

with st.expander("Предпросмотр (первые 5 строк)"):
    st.dataframe(raw_df.head(), use_container_width=True)

# ---- Запуск пайплайна ----
st.subheader("2. Запуск пайплайна")
if st.button("🔍 Найти дубликаты", type="primary"):
    # Подкладываем выбранные пороги
    import er.clustering as cl
    cl.THRESHOLD_AUTO_MERGE = auto_thr
    cl.THRESHOLD_REVIEW = review_thr

    progress = st.progress(0, "Старт")
    with st.spinner("Парсинг и схлопывание..."):
        progress.progress(20, "Блокинг...")
    result = run_inference(raw_df, model_path)
    progress.progress(100, "Готово")

    st.session_state["result"] = result
    st.session_state["raw_df"] = raw_df

if "result" not in st.session_state:
    st.info("Нажмите «Найти дубликаты», чтобы запустить пайплайн.")
    st.stop()

result = st.session_state["result"]
pairs = result["pairs_scored"].copy()
clusters = result["clusters"]
prof = result["prof"]
summary = result["summary"]

# ---- Сводка ----
st.subheader("3. Результаты")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Профилей", f"{summary['n_profiles']:,}")
c2.metric("Пар-кандидатов", f"{summary['n_pairs']:,}")
c3.metric("AUTO-MERGE", f"{summary['n_auto']:,}",
          help="Объединить автоматически — высокая уверенность")
c4.metric("REVIEW", f"{summary['n_review']:,}",
          help="Отправить на ручную проверку оператору")
c5.metric("Кластеров с дублями", f"{summary['n_clusters_with_dups']:,}")

# ---- Таблица пар ----
st.markdown("### 🔍 Найденные пары с вероятностями")
action_filter = st.multiselect(
    "Показывать действия:", ["AUTO_MERGE", "REVIEW", "KEEP_SEPARATE"],
    default=["AUTO_MERGE", "REVIEW"],
)
min_prob = st.slider("Минимальная вероятность", 0.0, 1.0, review_thr, 0.01)

shown = pairs[(pairs["action"].isin(action_filter)) & (pairs["prob_match"] >= min_prob)] \
    .sort_values("prob_match", ascending=False)

# Добавим характеристики профилей (для удобства просмотра)
display_cols = ["profile_id", "email", "phone", "first_name"]
present_display = [c for c in display_cols if c in prof.columns]
prof_idx = prof.set_index("profile_id")[present_display[1:]]


def _enrich(side: str) -> pd.DataFrame:
    return shown[[f"profile_id_{side}"]].rename(columns={f"profile_id_{side}": "profile_id"}) \
                                          .merge(prof_idx, left_on="profile_id", right_index=True, how="left") \
                                          .add_suffix(f"_{side}")


merged = shown.reset_index(drop=True)
merged = pd.concat([
    merged,
    _enrich("1").reset_index(drop=True).drop(columns=[f"profile_id_1"]),
    _enrich("2").reset_index(drop=True).drop(columns=[f"profile_id_2"]),
], axis=1)


def _color_action(val: str) -> str:
    return {"AUTO_MERGE": "background-color: #d4edda",
            "REVIEW":      "background-color: #fff3cd",
            "KEEP_SEPARATE": "background-color: #f8d7da"}.get(val, "")


styled = merged.head(500).style.format({"prob_match": "{:.3f}"}).map(_color_action, subset=["action"])
st.dataframe(styled, use_container_width=True, height=400)
st.caption(f"Показано {min(len(merged), 500)} из {len(merged):,} пар. AUTO-MERGE — зелёный, REVIEW — жёлтый.")

# ---- Кластеры ----
st.markdown("### 🧩 Кластеры дубликатов (после AUTO-MERGE)")
cluster_sizes = clusters.groupby("cluster_id").size().rename("size").reset_index()
multi_clusters = cluster_sizes[cluster_sizes["size"] > 1].sort_values("size", ascending=False)

if len(multi_clusters) == 0:
    st.info("Кластеров-дубликатов не найдено (при текущих порогах).")
else:
    sel = st.selectbox(
        "Выбери кластер для просмотра:",
        multi_clusters["cluster_id"].head(50).tolist(),
        format_func=lambda cid: f"Кластер #{cid} — {int(multi_clusters.loc[multi_clusters['cluster_id']==cid,'size'].iloc[0])} профилей",
    )
    members = clusters[clusters["cluster_id"] == sel]["profile_id"].tolist()
    member_df = prof[prof["profile_id"].isin(members)][present_display + ["created_at"]]
    st.dataframe(member_df, use_container_width=True)

    # Пары, удерживающие этот кластер
    pp = pairs[pairs["profile_id_1"].isin(members) & pairs["profile_id_2"].isin(members)] \
        .sort_values("prob_match", ascending=False)
    st.caption("Пары внутри кластера:")
    st.dataframe(pp[["profile_id_1", "profile_id_2", "prob_match", "action"]],
                 use_container_width=True, height=200)

# ---- Скачать ----
st.markdown("### 💾 Скачать результаты")
c1, c2 = st.columns(2)
with c1:
    buf = io.StringIO()
    pairs.sort_values("prob_match", ascending=False).to_csv(buf, index=False)
    st.download_button("⬇️ Пары (csv)", buf.getvalue(), "pairs.csv", "text/csv")
with c2:
    buf = io.StringIO()
    clusters.to_csv(buf, index=False)
    st.download_button("⬇️ Кластеры (csv)", buf.getvalue(), "clusters.csv", "text/csv")
