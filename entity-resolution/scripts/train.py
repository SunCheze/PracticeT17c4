"""CLI: обучение модели Entity Resolution.

Использование::

    # обучить на parquet с реальными данными
    python -m scripts.train --data data/raw/profiles.parquet

    # или на синтетике (для smoke-теста)
    python -m scripts.train --synthetic

Сохраняет модель и пороги в models/.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import sys

# Добавляем родительскую директорию в путь поиска
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import er
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report

from er.blocking import blocking_recall, label_pairs
from er.config import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLDS_PATH
from er.features import generate_pairwise_features
from er.model import (calibrate_thresholds, predict_proba, save_model,
                      save_thresholds, train_model)
from er.pipeline import generate_candidates, load_raw, preprocess
from er.synthetic import make_synthetic_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение модели Entity Resolution")
    p.add_argument("--data", type=Path, help="Путь к parquet/csv с сырыми данными")
    p.add_argument("--synthetic", action="store_true", help="Использовать синтетические данные")
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--thresholds-out", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--neg-ratio", type=int, default=30, help="Отношение негативов к позитивам в train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-precision-auto", type=float, default=0.99)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.data or args.synthetic):
        raise SystemExit("Укажи --data PATH или --synthetic")

    print("=== Загрузка данных ===")
    raw = make_synthetic_dataset(seed=args.seed) if args.synthetic else load_raw(args.data)
    print(f"  сырых строк: {len(raw):,}")

    print("=== Предобработка ===")
    prof = preprocess(raw)
    print(f"  профилей: {prof['profile_id'].nunique():,}  |  сущностей: {prof['entity_id'].nunique():,}")

    print("=== Блокинг ===")
    pairs = generate_candidates(prof, labeled=True)
    rec, cap, tot = blocking_recall(pairs, prof)
    print(f"  пар-кандидатов: {len(pairs):,}  |  blocking recall: {rec:.3f} ({cap}/{tot})")

    print("=== Сплит по entity_id ===")
    ents = np.asarray(prof[["profile_id", "entity_id"]].drop_duplicates()["entity_id"].unique())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ents)
    cut = int((1 - args.val_fraction) * len(ents))
    train_ents, val_ents = set(ents[:cut]), set(ents[cut:])
    train_profs = set(prof[prof["entity_id"].isin(train_ents)]["profile_id"])
    val_profs = set(prof[prof["entity_id"].isin(val_ents)]["profile_id"])

    train_pairs = pairs[pairs["profile_id_1"].isin(train_profs)
                        & pairs["profile_id_2"].isin(train_profs)].reset_index(drop=True)
    val_pairs = pairs[pairs["profile_id_1"].isin(val_profs)
                      & pairs["profile_id_2"].isin(val_profs)].reset_index(drop=True)

    pos = train_pairs[train_pairs["is_match"] == 1]
    neg = train_pairs[train_pairs["is_match"] == 0]
    neg_keep = min(len(neg), len(pos) * args.neg_ratio)
    train_bal = pd.concat([pos, neg.sample(neg_keep, random_state=args.seed)],
                          ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"  train: {len(train_bal):,} (поз {len(pos):,})  |  val: {len(val_pairs):,} (поз {(val_pairs['is_match']==1).sum()})")

    print("=== Признаки ===")
    Xt, cat_features = generate_pairwise_features(train_bal, prof)
    Xv, _ = generate_pairwise_features(val_pairs, prof)
    feat_cols = [c for c in Xt.columns if c not in ("profile_id_1", "profile_id_2", "is_match")]

    print("=== Обучение CatBoost ===")
    model = train_model(
        Xt[feat_cols], Xt["is_match"], cat_features,
        Xv[feat_cols], Xv["is_match"], random_seed=args.seed,
    )

    print("=== Оценка на val ===")
    prob = predict_proba(model, Xv[feat_cols], cat_features)
    pr_auc = average_precision_score(Xv["is_match"], prob)
    print(f"  PR-AUC = {pr_auc:.3f}")

    thresholds = calibrate_thresholds(Xv["is_match"].values, prob,
                                       target_precision_auto=args.target_precision_auto)
    thresholds["pr_auc"] = float(pr_auc)
    thresholds["blocking_recall"] = float(rec)
    print(f"  пороги: AUTO={thresholds['auto_merge']:.3f}  REVIEW={thresholds['review']:.3f}")

    pred_review = (prob >= thresholds["review"]).astype(int)
    print(classification_report(Xv["is_match"], pred_review,
                                target_names=["разные", "дубли"], digits=3))

    save_model(model, args.model_out)
    save_thresholds(thresholds, args.thresholds_out)
    print(f"  модель сохранена: {args.model_out}")
    print(f"  пороги сохранены: {args.thresholds_out}")


if __name__ == "__main__":
    main()
