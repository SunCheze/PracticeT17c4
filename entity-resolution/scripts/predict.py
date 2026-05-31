"""CLI: предсказание дубликатов на батче новых профилей.

Использование::

    python -m scripts.predict --input data/raw/new_batch.parquet \\
        --output-pairs data/processed/pairs.csv \\
        --output-clusters data/processed/clusters.csv

Модель и пороги читаются из models/ по умолчанию.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from er.config import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLDS_PATH
from er.pipeline import load_raw, run_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Поиск дубликатов на батче профилей")
    p.add_argument("--input", type=Path, required=True, help="parquet/csv с сырыми профилями")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-pairs", type=Path, default=Path("data/processed/pairs.csv"))
    p.add_argument("--output-clusters", type=Path, default=Path("data/processed/clusters.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"=== Чтение {args.input} ===")
    raw = load_raw(args.input)
    print(f"  строк: {len(raw):,}")

    print("=== Инференс ===")
    result = run_inference(raw, args.model)

    pairs = result["pairs_scored"]
    clusters = result["clusters"]
    summary = result["summary"]

    print(f"  профилей: {summary['n_profiles']:,}")
    print(f"  пар-кандидатов: {summary['n_pairs']:,}")
    print(f"  AUTO-MERGE: {summary['n_auto']:,}")
    print(f"  REVIEW: {summary['n_review']:,}")
    print(f"  кластеров с дублями: {summary['n_clusters_with_dups']:,}")

    args.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.sort_values("prob_match", ascending=False).to_csv(args.output_pairs, index=False)
    clusters.to_csv(args.output_clusters, index=False)
    print(f"  пары: {args.output_pairs}")
    print(f"  кластеры: {args.output_clusters}")


if __name__ == "__main__":
    main()
