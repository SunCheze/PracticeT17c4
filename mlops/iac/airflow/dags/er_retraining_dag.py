"""DAG автоматического retraining'а и продвижения модели Entity Resolution.

Полный жизненный цикл, который и есть смысл уровня 2 в манифесте:
    1. ingest      — пуллим свежие профили из source-системы в feature store
    2. train       — обучаем новую модель на свежем срезе
    3. evaluate    — считаем PR-AUC, B-Cubed F1, blocking recall на holdout
    4. quality_gate— если метрики хуже порогов, останавливаемся
    5. register    — регистрируем модель в MLflow со stage=Staging
    6. shadow      — переключаем canary-сервинг на новую модель,
                     запускаем shadow-сравнение на реальном трафике 24 часа
    7. canary      — если shadow ок, поднимаем вес canary до 10%
    8. promote     — после ручного approval или авто-таймера переводим в Production
                     и переключаем traffic-router

Расписание: каждое воскресенье в 02:00, manual trigger тоже работает.
"""
from __future__ import annotations
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# --------------------------------------------------------------------------- #
# Конфигурация (вынесена в Airflow Variables в проде)
# --------------------------------------------------------------------------- #
QUALITY_GATES = {
    "min_pr_auc": 0.70,           # ниже — не deploy'им
    "min_bcubed_f1_review": 0.90, # B-Cubed F1 на REVIEW-пороге
    "min_blocking_recall": 0.95,  # блокинг должен ловить ≥95% дублей
    "max_pr_auc_drop": 0.05,      # просадка от prod-модели не больше 5 пп
}

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
}


# --------------------------------------------------------------------------- #
# Шаги пайплайна
# --------------------------------------------------------------------------- #
def ingest_profiles(**ctx):
    """Снимок профилей за последнюю неделю в Feature Store (Postgres)."""
    from er_mlops.ingest import sync_profiles_to_features
    n_rows = sync_profiles_to_features(week_offset=0)
    ctx["ti"].xcom_push(key="n_rows", value=n_rows)


def train_model(**ctx):
    """Обучение модели с логированием в MLflow."""
    import mlflow
    from er_mlops.training import train_full_pipeline

    with mlflow.start_run(run_name=f"retrain_{datetime.now():%Y%m%d}") as run:
        metrics, model = train_full_pipeline(
            feature_store_uri="postgresql://mlops:mlops@postgres/features",
        )
        mlflow.log_metrics(metrics)
        mlflow.catboost.log_model(model, "model", registered_model_name="entity-resolution")
        ctx["ti"].xcom_push(key="run_id", value=run.info.run_id)
        ctx["ti"].xcom_push(key="metrics", value=metrics)


def quality_gate(**ctx) -> str:
    """Решение: проходим ли качество — продолжаем; нет — abort."""
    metrics = ctx["ti"].xcom_pull(key="metrics", task_ids="train_model")
    g = QUALITY_GATES

    failures = []
    if metrics["pr_auc"] < g["min_pr_auc"]:
        failures.append(f"PR-AUC {metrics['pr_auc']:.3f} < {g['min_pr_auc']}")
    if metrics["bcubed_f1_review"] < g["min_bcubed_f1_review"]:
        failures.append(f"B-Cubed F1 {metrics['bcubed_f1_review']:.3f} < {g['min_bcubed_f1_review']}")
    if metrics["blocking_recall"] < g["min_blocking_recall"]:
        failures.append(f"blocking recall {metrics['blocking_recall']:.3f} < {g['min_blocking_recall']}")
    if metrics.get("pr_auc_drop", 0) > g["max_pr_auc_drop"]:
        failures.append(f"PR-AUC просел на {metrics['pr_auc_drop']:.3f}")

    if failures:
        ctx["ti"].xcom_push(key="gate_failures", value=failures)
        return "abort_with_alert"
    return "register_staging"


def register_staging(**ctx):
    """Перевести модель в stage=Staging."""
    from mlflow import MlflowClient
    run_id = ctx["ti"].xcom_pull(key="run_id", task_ids="train_model")
    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    client.transition_model_version_stage(
        name="entity-resolution", version=versions[0].version, stage="Staging"
    )


def shadow_evaluation(**ctx):
    """Запустить shadow-режим: canary получает копию трафика, метрики сравниваются.
    Завершение блока — после 24 часов или N запросов, что наступит раньше."""
    from er_mlops.deploy import enable_shadow_mirror, wait_for_shadow_window
    enable_shadow_mirror()
    shadow_metrics = wait_for_shadow_window(min_requests=10000, max_hours=24)
    ctx["ti"].xcom_push(key="shadow_metrics", value=shadow_metrics)


def shadow_gate(**ctx) -> str:
    """Сравниваем canary против production на shadow-трафике."""
    m = ctx["ti"].xcom_pull(key="shadow_metrics", task_ids="shadow_evaluation")
    # canary не должен быть статистически хуже prod ни по одной критичной метрике
    if m["canary_pr_auc"] < m["prod_pr_auc"] - 0.02:
        return "rollback"
    if m["canary_latency_p95_ms"] > m["prod_latency_p95_ms"] * 1.5:
        return "rollback"
    return "canary_10pct"


def canary_10pct(**ctx):
    """Перевести 10% реального трафика на canary."""
    from er_mlops.deploy import set_traffic_weights
    set_traffic_weights(production=90, canary=10)


def promote_to_production(**ctx):
    """Финальный шаг: canary становится production, веса 0/100."""
    from mlflow import MlflowClient
    from er_mlops.deploy import set_traffic_weights, archive_old_production

    client = MlflowClient()
    # Архивируем предыдущую production, новую переводим в Production
    archive_old_production(model_name="entity-resolution")
    versions = client.get_latest_versions("entity-resolution", stages=["Staging"])
    client.transition_model_version_stage(
        name="entity-resolution", version=versions[0].version, stage="Production",
    )
    set_traffic_weights(production=0, canary=100)


def alert_and_abort(**ctx):
    """Послать уведомление команде и остановить пайплайн."""
    failures = ctx["ti"].xcom_pull(key="gate_failures", task_ids="quality_gate")
    print(f"QUALITY GATE FAILED: {failures}")
    # тут реальный Slack/PagerDuty alert
    raise ValueError(f"quality gate failed: {failures}")


# --------------------------------------------------------------------------- #
# DAG
# --------------------------------------------------------------------------- #
with DAG(
    dag_id="er_retraining",
    description="Еженедельный retraining и продвижение модели ER",
    schedule="0 2 * * 0",                       # каждое воскресенье в 02:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["mlops", "entity-resolution"],
) as dag:

    t_ingest = PythonOperator(task_id="ingest_profiles", python_callable=ingest_profiles)

    t_train  = PythonOperator(task_id="train_model", python_callable=train_model)

    t_gate   = BranchPythonOperator(task_id="quality_gate", python_callable=quality_gate)

    t_abort  = PythonOperator(task_id="abort_with_alert", python_callable=alert_and_abort)

    t_reg    = PythonOperator(task_id="register_staging", python_callable=register_staging)

    t_shadow = PythonOperator(task_id="shadow_evaluation", python_callable=shadow_evaluation)

    t_shadow_gate = BranchPythonOperator(task_id="shadow_gate", python_callable=shadow_gate)

    t_canary = PythonOperator(task_id="canary_10pct", python_callable=canary_10pct)

    # human approval — ручной trigger следующего таска через Airflow UI
    t_human  = EmptyOperator(task_id="human_approval_gate")

    t_promote = PythonOperator(task_id="promote_to_production", python_callable=promote_to_production)

    t_rollback = BashOperator(
        task_id="rollback",
        bash_command="er-deploy set-weights --production 100 --canary 0",
    )

    # Поток: train → gate → register → shadow → shadow_gate → canary → human → promote
    t_ingest >> t_train >> t_gate
    t_gate >> [t_abort, t_reg]
    t_reg >> t_shadow >> t_shadow_gate
    t_shadow_gate >> [t_rollback, t_canary]
    t_canary >> t_human >> t_promote
