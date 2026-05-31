# Entity Resolution — MLOps stack (Level 2)

Декларативное описание ML-инфраструктуры уровня 2.

## Состав

```
mlops/
├── manifest/                  # Шаг 1 — однострачный манифест зрелости
│   └── manifest.docx
├── iac/
│   ├── docker/                # Шаг 2 — основной артефакт инфраструктуры
│   │   ├── docker-compose.yml      # вся инфра в одном файле
│   │   ├── serving/                # FastAPI + CatBoost + Prometheus
│   │   ├── router/                 # Nginx с weighted routing
│   │   └── monitoring/             # Prometheus конфиг + алерты на SLO
│   ├── airflow/dags/          # Шаг 2 — оркестратор retraining'а
│   │   └── er_retraining_dag.py    # ingest→train→eval→shadow→canary→promote
│   └── terraform/             # Шаг 2 — cloud-вариант (AWS)
│       ├── main.tf                 # VPC, RDS, S3, ECS, ALB
│       └── variables.tf            # weight_production / weight_canary
└── docs/
    └── sli_slo.docx           # Шаг 3 — SLI/SLO для каждого компонента
```

## Запуск (on-prem)

```bash
cd iac/docker
docker compose up -d         # поднимет все 10 сервисов
```

UI доступны на:
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080
- Grafana: http://localhost:3000
- MinIO console: http://localhost:9001
- Сервинг (через router): http://localhost:80/score

## Запуск (cloud, AWS)

```bash
cd iac/terraform
terraform init
terraform plan -var-file=production.tfvars
terraform apply
```

## Переключение трафика

```bash
# Включить shadow-сравнение новой модели
terraform apply -var='weight_production=100' -var='weight_canary=0'
# (canary получает копию трафика через mirror, без ответа клиенту)

# Перейти в canary 10/90
terraform apply -var='weight_production=90' -var='weight_canary=10'

# Promote canary в production
terraform apply -var='weight_production=0' -var='weight_canary=100'

# Rollback
terraform apply -var='weight_production=100' -var='weight_canary=0'
```

В on-prem то же самое — через изменение `weight=` в `router/nginx.conf` и `nginx -s reload`.

## SLO

Полная таблица — в `docs/sli_slo.docx`. Главные:
- Сервинг p95 latency ≤ 100 мс, availability ≥ 99.9%
- Модель PR-AUC на shadow ≥ 0.70, FP rate в AUTO-MERGE ≤ 1%
- Retraining-DAG: успешность ≥ 95% за месяц, выполнение ≤ 4 часа
